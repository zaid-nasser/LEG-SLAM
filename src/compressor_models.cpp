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

#include "include/compressor_models.h"
#include <sstream>
using namespace lf_encoder;


TorchPCA::TorchPCA(
        int pixels_count,
        int input_feat_dim,
        int output_feat_dim,
        torch::DeviceType device_type)
        : PixelwiseCompressor(
              pixels_count,
              input_feat_dim,
              output_feat_dim) {

    if (device_type == torch::kCUDA &&
        torch::cuda::is_available()) {

        device_type_ = torch::kCUDA;

        std::cerr
            << "[TorchPCA] GPU is used."
            << std::endl;

    } else {

        device_type_ = torch::kCPU;

        std::cerr
            << "[TorchPCA] CPU is used."
            << std::endl;
    }
}

void TorchPCA::loadModel(
        const std::string &path_to_model) {

    try {

        module_ =
            std::make_unique<torch::jit::script::Module>(
                torch::jit::load(path_to_model)
            );

        module_->to(device_type_);
        module_->eval();

        model_enable_ = true;

    } catch (const std::exception &exc) {

        std::cerr
            << "[TorchPCA] "
            << exc.what()
            << std::endl;

        throw std::runtime_error(
            "Failed to load TorchScript PCA."
        );
    }
}

cv::Mat TorchPCA::predict(
        cv::Mat language_features) {

    if (!model_enable_) {
        throw std::runtime_error(
            "[TorchPCA] Compressor is not initialized."
        );
    }


    if (language_features.cols != input_feat_dim_) {

        std::stringstream message;

        message
            << "[TorchPCA] Expected feature dimension "
            << input_feat_dim_
            << ", got "
            << language_features.cols;

        throw std::runtime_error(message.str());
    }


    if (!language_features.isContinuous()) {
        language_features =
            language_features.clone();
    }


    // OpenCV [N,1152] -> Torch [N,1152]

    auto input_tensor =
        torch::from_blob(
            language_features.ptr<float>(),
            {
                language_features.rows,
                language_features.cols
            },
            torch::TensorOptions().dtype(torch::kFloat32)
        )
        .clone()
        .to(device_type_);


    torch::NoGradGuard no_grad;


    std::vector<torch::jit::IValue> inputs;

    inputs.emplace_back(input_tensor);

    // K, e.g. 64
    inputs.emplace_back(
        static_cast<int64_t>(output_feat_dim_)
    );


    auto output =
        module_->forward(inputs)
               .toTensor();


    if (output.dim() != 2 ||
        output.size(0) != language_features.rows ||
        output.size(1) != output_feat_dim_) {

        throw std::runtime_error(
            "[TorchPCA] Unexpected PCA output shape."
        );
    }


    output =
        output.detach()
              .to(torch::kCPU)
              .to(torch::kFloat32)
              .contiguous();


    cv::Mat result(
        output.size(0),
        output.size(1),
        CV_32F,
        output.data_ptr<float>()
    );


    return result.clone();
}
