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

#include "include/language_features_encoder.h"
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace lf_encoder;


std::tuple<LanguageFeaturesEncoder::EncoderParams, LanguageFeaturesEncoder::CompressorParams>
LanguageFeaturesEncoder::readConfigFromRegistry(
    const std::filesystem::path &registry_path,
    const std::string &feature_model_id) {
    std::ifstream registry_stream(registry_path);
    if (!registry_stream.is_open()) {
        std::stringstream message_stream;
        message_stream << "Failed to open feature model registry at: " << registry_path;
        throw std::runtime_error(message_stream.str());
    }

    Json::Value registry_root;
    registry_stream >> registry_root;
    const Json::Value models = registry_root["models"];
    if (!models.isObject() || !models.isMember(feature_model_id)) {
        std::stringstream message_stream;
        message_stream << "Model '" << feature_model_id << "' is not configured in registry: " << registry_path;
        throw std::runtime_error(message_stream.str());
    }

    const Json::Value model_root = models[feature_model_id];
    const Json::Value encoder_root = model_root["encoder"];
    const Json::Value compressor_root = model_root["compressor"];
    if (!encoder_root.isObject() || !compressor_root.isObject()) {
        throw std::runtime_error("Invalid registry format: encoder/compressor sections are missing.");
    }

    auto encoder_type = encoder_root["type"].asString();
    auto path_to_encoder_model = encoder_root["path"].asString();
    auto encoder_input_width = encoder_root["input_width"].asInt();
    auto encoder_input_height = encoder_root["input_height"].asInt();
    auto encoder_output_pixels = encoder_root["output_pixels"].asInt();
    auto encoder_output_feat_dim = encoder_root["embedding_size"].asInt();

    auto compressor_type = compressor_root["type"].asString();
    auto path_to_compressor_model = compressor_root["path"].asString();
    auto compress_embeddings_size = compressor_root["output_dim"].asInt();

    int resolution_step = 0;

    if (encoder_root.isMember("resolution_step")) {
        resolution_step =
            encoder_root["resolution_step"].asInt();
    }

    return {{encoder_type, path_to_encoder_model, encoder_input_width, encoder_input_height, encoder_output_pixels, encoder_output_feat_dim,resolution_step},
            {compressor_type, path_to_compressor_model, encoder_output_pixels, encoder_output_feat_dim, compress_embeddings_size}};
}

LanguageFeaturesEncoder::LanguageFeaturesEncoder(
    std::filesystem::path model_registry_path,
    torch::DeviceType device_type,
    std::optional<std::string> feature_model_id)
    :
    feature_model_id_(std::move(feature_model_id)),
    model_registry_path_(std::move(model_registry_path)) {
//    device_type_ = torch::cuda::is_available() ? device_type : torch::kCPU;
    if (torch::cuda::is_available() && device_type == torch::kCUDA) {  // TODO: It is necessary to provide automatic selection of the memory size.
        c10::cuda::CUDACachingAllocator::init(1);
        c10::cuda::CUDACachingAllocator::setMemoryFraction(0.8, 0);
        device_type_ = torch::kCUDA;
    } else {
        device_type_ = torch::kCPU;
    }

    auto [encoder_config, compressor_config] = readConfigFromRegistry(
        model_registry_path_, feature_model_id_.value_or("radio_v4_so400m"));

    if (encoder_config.type != "RadioTorchScript") {
        throw std::runtime_error("Unsupported encoder type: " + encoder_config.type);
    }
    if (encoder_config.resolution_step <= 0 || encoder_config.output_feat_dim <= 0) {
        throw std::runtime_error("Invalid RADIO resolution step or embedding dimension.");
    }
    if (compressor_config.type != "TorchPCA") {
        throw std::runtime_error("Unsupported compressor type: " + compressor_config.type);
    }
    if (compressor_config.output_feat_dim <= 0 ||
        compressor_config.output_feat_dim > compressor_config.input_feat_dim) {
        throw std::runtime_error("Invalid PCA output dimension.");
    }

    encoder_ = std::make_unique<RadioTorchScriptEncoder>(
        encoder_config.output_feat_dim, encoder_config.resolution_step, device_type_);
    encoder_->loadModel(encoder_config.path);

    compressor_ = std::make_unique<TorchPCA>(
        compressor_config.pixels_count, compressor_config.input_feat_dim,
        compressor_config.output_feat_dim, device_type_);
    compressor_->loadModel(compressor_config.path);


}

cv::Mat LanguageFeaturesEncoder::createLanguageFeatures(const cv::Mat &rgb_image) {

    auto encoder_output = encoder_->predict(rgb_image);

    auto compressed_language_features =
        compressor_->predict(encoder_output.features);

    const int expected_pixels =
        encoder_output.height * encoder_output.width;

    if (compressed_language_features.rows != expected_pixels) {
        std::stringstream message;
        message
            << "[LanguageFeaturesEncoder] Invalid number of feature pixels. "
            << "Expected "
            << encoder_output.height << " x "
            << encoder_output.width << " = "
            << expected_pixels
            << ", but compressor returned "
            << compressed_language_features.rows;

        throw std::runtime_error(message.str());
    }

    auto language_features_image =
        compressed_language_features.reshape(
            compressed_language_features.cols,
            encoder_output.height
        );

    if (language_features_image.cols != encoder_output.width) {
        throw std::runtime_error(
            "[LanguageFeaturesEncoder] Invalid reshaped feature-map width."
        );
    }

    std::cout
        << "[LF] raw="
        << encoder_output.features.rows
        << "x"
        << encoder_output.features.cols
        << " grid="
        << encoder_output.height
        << "x"
        << encoder_output.width
        << " compressed="
        << language_features_image.rows
        << "x"
        << language_features_image.cols
        << "x"
        << language_features_image.channels()
        << std::endl;

    return language_features_image;
}
