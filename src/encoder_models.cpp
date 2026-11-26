/**
 * This file is part of LEGS-SLAM
 *
 * Copyright (C) 2024-2025 Egor Zubkov and Roman Titkov, Center for Cognitive Modeling MIPT
 *
 * LEGS-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LEGS-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with LEGS-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/passes/autocast.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <cstdlib>
#include <string>
#include "include/encoder_models.h"
#include <cmath>
#include <sstream>
#include <stdexcept>
using namespace lf_encoder;


namespace {

class RadioAutocastGuard {
public:
    explicit RadioAutocastGuard(at::ScalarType dtype)
        : prev_enabled_(
              at::autocast::is_autocast_enabled(at::kCUDA)),
          prev_dtype_(
              at::autocast::get_autocast_dtype(at::kCUDA)),
          prev_cache_(
              at::autocast::is_autocast_cache_enabled()) {

        const bool enabled = dtype != at::kFloat;
        if (enabled) {
            at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        }
        at::autocast::set_autocast_enabled(at::kCUDA, enabled);

        at::autocast::set_autocast_cache_enabled(false);
        at::autocast::increment_nesting();
    }

    ~RadioAutocastGuard() {
        at::autocast::set_autocast_enabled(
            at::kCUDA, prev_enabled_
        );

        at::autocast::set_autocast_dtype(
            at::kCUDA, prev_dtype_
        );

        if (at::autocast::decrement_nesting() == 0) {
            at::autocast::clear_cache();
        }

        at::autocast::set_autocast_cache_enabled(
            prev_cache_
        );
    }

    RadioAutocastGuard(const RadioAutocastGuard&) = delete;
    RadioAutocastGuard& operator=(
        const RadioAutocastGuard&) = delete;

private:
    bool prev_enabled_;
    at::ScalarType prev_dtype_;
    bool prev_cache_;
};

at::ScalarType selectRadioPrecision(torch::DeviceType device) {
    const char* value = std::getenv("LEGS_RADIO_PRECISION");
    const std::string requested = value ? value : "auto";

    if (requested != "auto" && requested != "fp32" &&
        requested != "fp16" && requested != "bf16") {
        throw std::invalid_argument(
            "LEGS_RADIO_PRECISION must be auto, fp32, fp16 or bf16; got: " + requested);
    }

    if (requested == "fp32") {
        return at::kFloat;
    }
    if (device != torch::kCUDA) {
        if (requested != "auto") {
            throw std::invalid_argument("RADIO mixed precision requires a CUDA device.");
        }
        return at::kFloat;
    }

    const int major = at::cuda::getCurrentDeviceProperties()->major;
    if (requested == "auto") {
        // Keep FP32 on pre-Ampere GPUs until FP16 has been validated there.
        return major >= 8 ? at::kBFloat16 : at::kFloat;
    }
    if (requested == "bf16") {
        if (major < 8) {
            throw std::invalid_argument("RADIO BF16 requires an Ampere or newer CUDA GPU.");
        }
        return at::kBFloat16;
    }
    return at::kHalf;
}

const char* radioPrecisionName(at::ScalarType dtype) {
    switch (dtype) {
        case at::kBFloat16: return "BF16 autocast";
        case at::kHalf: return "FP16 autocast";
        default: return "FP32";
    }
}

} // anonymous namespace
RadioTorchScriptEncoder::RadioTorchScriptEncoder(
        int output_feat_dim,
        int resolution_step,
        torch::DeviceType device_type)
        : LFEncoder(
              0,                  // input width is dynamic
              0,                  // input height is dynamic
              0,                  // number of patches is dynamic
              output_feat_dim),
          resolution_step_(resolution_step) {

    if (device_type == torch::kCUDA &&
        torch::cuda::is_available()) {

        device_type_ = torch::kCUDA;

        std::cerr
            << "[RadioTorchScriptEncoder] GPU is used."
            << std::endl;

    } else {

        device_type_ = torch::kCPU;

        std::cerr
            << "[RadioTorchScriptEncoder] CPU is used."
            << std::endl;
    }
    inference_dtype_ = selectRadioPrecision(device_type_);
}

int RadioTorchScriptEncoder::getNearestSupportedResolution(
        int size) const {

    const int step = resolution_step_;

    const int q = size / step;
    const int r = size % step;

    int rounded_q;

    if (2 * r < step) {

        rounded_q = q;

    } else if (2 * r > step) {

        rounded_q = q + 1;

    } else {

        // Python round() uses ties-to-even.
        //
        // Example:
        // 680 / 16 = 42.5
        // round(42.5) = 42
        // 42 * 16 = 672

        rounded_q =
            (q % 2 == 0)
            ? q
            : q + 1;
    }

    return std::max(
        rounded_q * step,
        step
    );
}

void RadioTorchScriptEncoder::loadModel(
        const std::string &path_to_model) {
    try {
        torch::jit::setGraphExecutorOptimize(false);
        torch::jit::setAutocastMode(false);
        module_ =
            std::make_unique<torch::jit::script::Module>(
                torch::jit::load(path_to_model)
            );

        module_->to(device_type_);
        module_->eval();
        std::cerr
            << "[RadioTorchScriptEncoder] Inference precision: "
            << radioPrecisionName(inference_dtype_)
            << std::endl;

        model_enable_ = true;

        std::cerr
            << "[RadioTorchScriptEncoder] Loaded model: "
            << path_to_model
            << std::endl;

    } catch (const std::exception &exc) {

        std::cerr
            << "[RadioTorchScriptEncoder] "
            << exc.what()
            << std::endl;

        throw std::runtime_error(
            "Failed to load RADIO TorchScript model."
        );
    }
}

LFEncoderOutput RadioTorchScriptEncoder::predict(
        const cv::Mat &image) {

    if (!model_enable_) {
        throw std::runtime_error(
            "[RadioTorchScriptEncoder] Model is not initialized."
        );
    }

    if (image.channels() != 3) {
        throw std::runtime_error(
            "[RadioTorchScriptEncoder] Expected 3-channel RGB image."
        );
    }


    // ----------------------------------------------------
    // 1. HWC uint8 OpenCV image -> Torch tensor
    // ----------------------------------------------------

    cv::Mat continuous_image =
        image.isContinuous()
        ? image
        : image.clone();

    auto x = torch::from_blob(
        continuous_image.data,
        {
            1,
            continuous_image.rows,
            continuous_image.cols,
            3
        },
        torch::TensorOptions().dtype(torch::kUInt8)
    );

    // [1,H,W,3] -> [1,3,H,W]
    x = x.permute({0, 3, 1, 2})
         .to(torch::kFloat32)
         .div(255.0)
         .to(device_type_);


    // ----------------------------------------------------
    // 2. RADIO-supported resolution
    // ----------------------------------------------------

    const int input_height =
        getNearestSupportedResolution(image.rows);

    const int input_width =
        getNearestSupportedResolution(image.cols);


    if (input_height != image.rows ||
        input_width != image.cols) {

        x = torch::nn::functional::interpolate(
            x,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{
                    input_height,
                    input_width
                })
                .mode(torch::kBilinear)
                .align_corners(false)
                .antialias(true)
        );
    }


    // ----------------------------------------------------
    // 3. RADIO forward
    // ----------------------------------------------------

    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x);

    torch::Tensor output;

    {
        // Disable graph optimization in the actual
        // inference thread.
        torch::jit::GraphOptimizerEnabledGuard jit_guard(false);

        if (device_type_ == torch::kCUDA) {

            RadioAutocastGuard autocast_guard(inference_dtype_);

            output = module_->forward(inputs).toTensor();

        } else {

            output = module_->forward(inputs).toTensor();
        }
    }

    // Preserve the original FP32 feature interface.
    output = output.to(torch::kFloat32);


    // Expected:
    // [1, N, 1152]

    if (output.dim() != 3 ||
        output.size(0) != 1) {

        throw std::runtime_error(
            "[RadioTorchScriptEncoder] "
            "Expected output shape [1,N,D]."
        );
    }


    const int feature_height =
        input_height / resolution_step_;

    const int feature_width =
        input_width / resolution_step_;

    const int expected_pixels =
        feature_height * feature_width;


    if (output.size(1) != expected_pixels) {

        std::stringstream message;

        message
            << "[RadioTorchScriptEncoder] "
            << "Unexpected feature count. Expected "
            << feature_height << " x "
            << feature_width << " = "
            << expected_pixels
            << ", got "
            << output.size(1);

        throw std::runtime_error(message.str());
    }


    if (output.size(2) != output_feat_dim_) {

        std::stringstream message;

        message
            << "[RadioTorchScriptEncoder] "
            << "Expected embedding dimension "
            << output_feat_dim_
            << ", got "
            << output.size(2);

        throw std::runtime_error(message.str());
    }


    // ----------------------------------------------------
    // 4. [1,N,D] -> CPU cv::Mat [N,D]
    // ----------------------------------------------------

    auto features =
        output.squeeze(0)
              .detach()
              .to(torch::kCPU)
              .to(torch::kFloat32)
              .contiguous();

    cv::Mat features_cv(
        features.size(0),
        features.size(1),
        CV_32F,
        features.data_ptr<float>()
    );


    // clone() is important because `features`
    // disappears when this function returns.

    return {
        features_cv.clone(),
        feature_height,
        feature_width
    };
}