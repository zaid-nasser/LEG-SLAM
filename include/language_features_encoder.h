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

#ifndef LANGUAGE_FEATURES_ENCODER_H
#define LANGUAGE_FEATURES_ENCODER_H

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>


#include <jsoncpp/json/json.h>

#include "include/compressor_models.h"
#include "include/encoder_models.h"


namespace lf_encoder {
    class LanguageFeaturesEncoder {
    private:
        struct EncoderParams {
            std::string type, path;

            int input_width;
            int input_height;

            int output_pixels_count;
            int output_feat_dim;

            int resolution_step = 0;
        };

        struct CompressorParams {
            std::string type, path;
            int pixels_count, input_feat_dim, output_feat_dim;
        };

        static std::tuple<EncoderParams, CompressorParams> readConfigFromRegistry(
                const std::filesystem::path &registry_path,
                const std::string &feature_model_id);
    public:
        explicit LanguageFeaturesEncoder(
                std::filesystem::path model_registry_path,
                torch::DeviceType device_type = torch::kCPU,
                std::optional<std::string> feature_model_id = std::nullopt);
        cv::Mat createLanguageFeatures(const cv::Mat &rgb_image);
        // cv::Mat createLanguageFeatures(const cv::Mat &rgb_image, const std::string &image_name);
    private:
        std::optional<std::string> feature_model_id_;
        std::filesystem::path model_registry_path_;
        torch::DeviceType device_type_;
        // Models
        std::unique_ptr<LFEncoder> encoder_;
        std::unique_ptr<PixelwiseCompressor> compressor_;
    };
}

#endif // LANGUAGE_FEATURES_ENCODER_H
