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

#ifndef ENCODER_MODELS_H
#define ENCODER_MODELS_H

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
//#include <tensorflow/cc/client/client_session.h>
//#include <tensorflow/cc/ops/io_ops.h>
//#include <tensorflow/core/framework/device_factory.h>
//#include <tensorflow/core/framework/tensor.h>
//#include <tensorflow/core/util/port.h>
//#include <tensorflow_cpp/model.h>


#include "include/tensor_utils.h"


namespace lf_encoder {

    struct LFEncoderOutput {
    cv::Mat features;   // [N, D]
    int height;         // feature grid height
    int width;          // feature grid width
        };

    class LFEncoder {
    public:
        explicit LFEncoder(int input_width, int input_height, int output_pixels_count, int output_feat_dim) :
                input_width_(input_width), input_height_(input_height),
                output_pixels_count_(output_pixels_count), output_feat_dim_(output_feat_dim) {}
        virtual ~LFEncoder() = default;
        virtual void loadModel(const std::string &path_to_model) = 0;
        virtual LFEncoderOutput predict(const cv::Mat &image) = 0;

    protected:
        int input_width_;
        int input_height_;
        int output_pixels_count_;
        int output_feat_dim_;
    };



    class RadioTorchScriptEncoder : public LFEncoder {
    public:
        explicit RadioTorchScriptEncoder(
            int output_feat_dim,
            int resolution_step,
            torch::DeviceType device_type);

        void loadModel(const std::string &path_to_model) override;

        LFEncoderOutput predict(const cv::Mat &image) override;

    private:
        int getNearestSupportedResolution(int size) const;

        int resolution_step_;
        torch::DeviceType device_type_;
        at::ScalarType inference_dtype_ = at::kFloat;

        bool model_enable_ = false;

        std::unique_ptr<torch::jit::script::Module> module_;
    };

}

#endif // ENCODER_MODELS_H
