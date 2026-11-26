/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <optional>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <regex>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/language_features_encoder.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

enum class DatasetType {
    Replica,
    ScanNet,
    MMPR
};

void LoadImagesReplica(const std::filesystem::path &pathDir, std::vector <std::string> &vstrImageFilenamesRGB,
                std::vector <std::string> &vstrImageFilenamesD);

void LoadImagesScanNet(const std::filesystem::path &pathDir, std::vector<std::string> &vstrImageFilenamesRGB,
                       std::vector<std::string> &vstrImageFilenamesD);

void LoadImagesMMPR(const std::filesystem::path &pathDir,
                    std::vector<std::string> &vstrImageFilenamesRGB,
                    std::vector<std::string> &vstrImageFilenamesD);

cv::Mat LoadDepthNPY(const std::filesystem::path &filename);

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

int main(int argc, char **argv) {
    std::vector<std::string> positional_args;
    bool use_viewer = true;
    std::optional<DatasetType> dataset_type;
    std::optional<std::string> feature_model_id;
    for (int idx = 1; idx < argc; ++idx) {
        std::string arg = argv[idx];
        if (arg == "--feature-model") {
            if (idx + 1 >= argc) {
                std::cerr << "--feature-model requires a value." << std::endl;
                return 1;
            }

            feature_model_id = std::string(argv[++idx]);
        }
        else if (arg == "--dataset") {
            if (idx + 1 >= argc) {
                std::cerr << "--dataset requires a value." << std::endl;
                return 1;
            }

            const std::string value = argv[++idx];

            if (value == "replica") {
                dataset_type = DatasetType::Replica;
            }
            else if (value == "scannet") {
                dataset_type = DatasetType::ScanNet;
            }
            else if (value == "mmpr") {
                dataset_type = DatasetType::MMPR;
            }
            else {
                std::cerr
                    << "Unsupported dataset: "
                    << value
                    << ". Expected replica, scannet, or mmpr."
                    << std::endl;
                return 1;
            }
        }
        else if (arg == "no_viewer") {
            use_viewer = false;
        }
        else {
            positional_args.emplace_back(arg);
        }
    }

    if (!dataset_type.has_value()) {
        std::cerr
            << "--dataset is required. Expected one of: replica, scannet, mmpr."
            << std::endl;
        return 1;
    }

    if (positional_args.size() != 6 && positional_args.size() != 7) {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"
                  << " path_to_ORB_SLAM3_settings"
                  << " path_to_feature_model_registry"
                  << " path_to_gaussian_mapping_settings"
                  << " path_to_sequence"
                  << " path_to_output_directory"
                  << " (optional)shutdown_ms"
                  << " (optional)no_viewer"
                  << " --dataset replica|scannet|mmpr"
                  << " (optional) --feature-model "
                     "radio_v4_so400m"
                  << std::endl;
        return 1;
    }

    std::string output_directory = positional_args[5];
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Retrieve paths to images
    std::vector <std::string> vstrImageFilenamesRGB;
    std::vector <std::string> vstrImageFilenamesD;
    std::string strImageDir = positional_args[4];
    int shutdown_ms = positional_args.size() == 7 ? std::stoi(positional_args[6]) : 0;
    std::filesystem::path pathImageDir(strImageDir);
    switch (*dataset_type) {
        case DatasetType::Replica:
            LoadImagesReplica(
                pathImageDir,
                vstrImageFilenamesRGB,
                vstrImageFilenamesD
            );
            break;

        case DatasetType::ScanNet:
            LoadImagesScanNet(
                pathImageDir,
                vstrImageFilenamesRGB,
                vstrImageFilenamesD
            );
            break;

        case DatasetType::MMPR:
            LoadImagesMMPR(
                pathImageDir,
                vstrImageFilenamesRGB,
                vstrImageFilenamesD
            );
            break;
    }

    // Check consistency in the number of images
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty()) {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr <ORB_SLAM3::System> pSLAM =
            std::make_shared<ORB_SLAM3::System>(
                    positional_args[0], positional_args[1], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

    // Create LanguageFeaturesEncoder
    std::filesystem::path feature_registry_path(positional_args[2]);
    std::shared_ptr<lf_encoder::LanguageFeaturesEncoder> pLFEncoder =
            std::make_shared<lf_encoder::LanguageFeaturesEncoder>(
                    feature_registry_path, device_type, feature_model_id);
    pSLAM->SetLanguageFeatureExtractor(
        [pLFEncoder](const cv::Mat& image)
        {
            return pLFEncoder->createLanguageFeatures(image);
        }
    );

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(positional_args[3]);
    std::shared_ptr <GaussianMapper> pGausMapper =
            std::make_shared<GaussianMapper>(
                    pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr <ImGuiViewer> pViewer;
    if (use_viewer) {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

    const int imageWidth = pSLAM->getSettings()->newImSize().width;
    const int imageHeight = pSLAM->getSettings()->newImSize().height;
    // Main loop
    cv::Mat imRGB, imD;
    double mean_frame_time = 0.0;
    double mean_track_time = 0.0;
    double mean_load_time = 0.0;

    int processed_images = 0;

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    for (int ni = 0; ni < nImages; ni++) {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        if (ni % 100 == 0) {
            std::cout << "Processing image: " << ni + 1 << " / " << nImages << std::endl;
        }

        if (pSLAM->isShutDown())
            break;

        // Benchmark image loading
        std::chrono::steady_clock::time_point t_load_start = std::chrono::steady_clock::now();
        
        // Read image and depthmap from file
        // Read RGB image and depth map.
        imRGB = cv::imread(
            vstrImageFilenamesRGB[ni],
            cv::IMREAD_UNCHANGED
        );

        // MMPR stores metric float32 depth in NumPy .npy files.
        // Replica and ScanNet continue to use their original image readers.
        if (*dataset_type == DatasetType::MMPR) {
            imD = LoadDepthNPY(vstrImageFilenamesD[ni]);
        } else {
            imD = cv::imread(
                vstrImageFilenamesD[ni],
                cv::IMREAD_UNCHANGED
            );
        }

        if (imRGB.empty()) {
            std::cerr
                << "Failed to load RGB image: "
                << vstrImageFilenamesRGB[ni]
                << std::endl;
            return 1;
        }

        if (imD.empty()) {
            std::cerr
                << "Failed to load depth image: "
                << vstrImageFilenamesD[ni]
                << std::endl;
            return 1;
        }

        if (ni == 0) {
            std::cout
                << "[Input] Raw RGB resolution: "
                << imRGB.cols << "x" << imRGB.rows
                << std::endl;

            std::cout
                << "[Input] Raw depth resolution: "
                << imD.cols << "x" << imD.rows
                << std::endl;

            std::cout
                << "[Input] ORB newImSize: "
                << imageWidth << "x" << imageHeight
                << std::endl;

            std::cout
                << "[Input] ORB imageScale: "
                << imageScale
                << std::endl;
        }

        cv::cvtColor(
            imRGB,
            imRGB,
            cv::COLOR_BGR2RGB
        );

        if (imRGB.size() != cv::Size(imageWidth, imageHeight)) {
            cv::resize(
                imRGB,
                imRGB,
                cv::Size(imageWidth, imageHeight)
            );
        }

        if (imD.size() != cv::Size(imageWidth, imageHeight)) {
            cv::resize(
                imD,
                imD,
                cv::Size(imageWidth, imageHeight),
                0, 0,
                (*dataset_type == DatasetType::MMPR)
                    ? cv::INTER_NEAREST : cv::INTER_LINEAR
            );
        }

        if (imageScale != 1.f) {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;

            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(
                imD, imD, cv::Size(width, height), 0, 0,
                (*dataset_type == DatasetType::MMPR)
                    ? cv::INTER_NEAREST : cv::INTER_LINEAR
            );
        }

        if (ni == 0) {
            std::cout
                << "[Input] Effective RGB resolution: "
                << imRGB.cols << "x" << imRGB.rows
                << std::endl;

            std::cout
                << "[Input] Effective depth resolution: "
                << imD.cols << "x" << imD.rows
                << std::endl;
        }
        double tframe = ni;
        std::chrono::steady_clock::time_point t_load_end = std::chrono::steady_clock::now();
        double t_load = std::chrono::duration_cast<std::chrono::duration<double>>(t_load_end - t_load_start).count();
        mean_load_time += t_load;

        // Benchmark tracking
        std::chrono::steady_clock::time_point t_track_start = std::chrono::steady_clock::now();
        pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);
        std::chrono::steady_clock::time_point t_track_end = std::chrono::steady_clock::now();
        double t_track = std::chrono::duration_cast<std::chrono::duration<double>>(t_track_end - t_track_start).count();
        mean_track_time += t_track;
        vTimesTrack[ni] = static_cast<float>(t_track);

        // Total frame time
        std::chrono::steady_clock::time_point t_frame_end = std::chrono::steady_clock::now();
        double t_frame = std::chrono::duration_cast<std::chrono::duration<double>>(t_frame_end - t0).count();
        mean_frame_time += t_frame;
        
        processed_images++;

        if (ni % 100 == 0) {
            std::cout
                << "Load time: "
                << t_load * 1000.0
                << " ms (mean: "
                << mean_load_time * 1000.0 / processed_images
                << " ms)"
                << std::endl;

            std::cout
                << "Track time: "
                << t_track * 1000.0
                << " ms (mean: "
                << mean_track_time * 1000.0 / processed_images
                << " ms)"
                << std::endl;

            std::cout
                << "Frame time: "
                << t_frame * 1000.0
                << " ms (mean: "
                << mean_frame_time * 1000.0 / processed_images
                << " ms)"
                << std::endl;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();

    const double t_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            t_end - t_start
        ).count();

    std::cout
        << "Total processing time: "
        << t_total
        << " s"
        << std::endl;

    if (processed_images > 0) {
        const double average_frame_ms =
            (t_total / processed_images) * 1000.0;

        const double average_fps =
            processed_images / t_total;

        std::cout
            << "Average time per image: "
            << average_frame_ms
            << " ms"
            << std::endl;

        std::cout
            << "Average FPS: "
            << average_fps
            << std::endl;
    }

    // Stop all threads
    std::cout << "Waiting " << shutdown_ms << " ms before shutdown..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(shutdown_ms));
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    vTimesTrack.resize(processed_images);
    saveTrackingTime (vTimesTrack,(output_dir / "TrackingTime.txt").string());


    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}

void LoadImagesReplica(const std::filesystem::path &pathDir, std::vector <std::string> &vstrImageFilenamesRGB,
                std::vector <std::string> &vstrImageFilenamesD) {
    std::filesystem::path pathImageDir = pathDir / "results";
    for (const auto &imagePath: std::filesystem::directory_iterator(pathImageDir)) {
        std::string name = imagePath.path().filename().string();
        if (name.rfind("frame", 0) == 0)
            vstrImageFilenamesRGB.push_back(imagePath.path().string());
        else if (name.rfind("depth", 0) == 0)
            vstrImageFilenamesD.push_back(imagePath.path().string());
        std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end());
        std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end());
    }
}

void LoadImagesScanNet(const std::filesystem::path &pathDir, std::vector<std::string> &vstrImageFilenamesRGB,
                       std::vector<std::string> &vstrImageFilenamesD) {
    std::filesystem::path colorDir = pathDir / "color";
    std::filesystem::path depthDir = pathDir / "depth";

    for (const auto &imagePath : std::filesystem::directory_iterator(colorDir)) {
        vstrImageFilenamesRGB.push_back(imagePath.path().string());
    }

    for (const auto &imagePath : std::filesystem::directory_iterator(depthDir)) {
        vstrImageFilenamesD.push_back(imagePath.path().string());
    }
    auto sortByValue = [](const std::string &a, const std::string &b) {
        int numA = std::stoi(a.substr(a.find_last_of('/') + 1));
        int numB = std::stoi(b.substr(b.find_last_of('/') + 1));
        return numA < numB;
    };

    std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end(), sortByValue);
    std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end(), sortByValue);
}

// Discover MMPR RGB-D frames, pairing files by their exact timestamp stems.
// The filenames are sorted numerically; cam1/cam2/cam3 and trajectories are ignored.
void LoadImagesMMPR(const std::filesystem::path &pathDir,
                    std::vector<std::string> &vstrImageFilenamesRGB,
                    std::vector<std::string> &vstrImageFilenamesD) {
    const auto colorDir = pathDir / "cam0" / "data";
    const auto depthDir = pathDir / "depth" / "data";

    if (!std::filesystem::is_directory(colorDir) ||
        !std::filesystem::is_directory(depthDir)) {
        throw std::runtime_error(
            "MMPR requires cam0/data/ and depth/data/ under " + pathDir.string());
    }

    std::map<std::string, std::string> rgbByTimestamp;
    std::map<std::string, std::string> depthByTimestamp;

    for (const auto &entry : std::filesystem::directory_iterator(colorDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;
        const auto timestamp = entry.path().stem().string();
        if (timestamp.empty() || !std::all_of(timestamp.begin(), timestamp.end(),
              [](unsigned char ch) { return ch >= '0' && ch <= '9'; }))
            continue;
        rgbByTimestamp.emplace(timestamp, entry.path().string());
    }
    for (const auto &entry : std::filesystem::directory_iterator(depthDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".npy")
            continue;
        const auto timestamp = entry.path().stem().string();
        if (timestamp.empty() || !std::all_of(timestamp.begin(), timestamp.end(),
              [](unsigned char ch) { return ch >= '0' && ch <= '9'; }))
            continue;
        depthByTimestamp.emplace(timestamp, entry.path().string());
    }

    if (rgbByTimestamp.empty() || depthByTimestamp.empty())
        throw std::runtime_error("MMPR has no timestamped RGB PNGs or depth NPYs: " +
                                 pathDir.string());
    if (rgbByTimestamp.size() != depthByTimestamp.size())
        throw std::runtime_error("MMPR RGB/depth file counts do not match: " +
                                 std::to_string(rgbByTimestamp.size()) + " RGB, " +
                                 std::to_string(depthByTimestamp.size()) + " depth");

    std::vector<std::string> timestamps;
    timestamps.reserve(rgbByTimestamp.size());
    for (const auto &[timestamp, rgbPath] : rgbByTimestamp) {
        if (depthByTimestamp.find(timestamp) == depthByTimestamp.end())
            throw std::runtime_error("MMPR is missing depth for timestamp " + timestamp);
        timestamps.push_back(timestamp);
    }
    std::sort(timestamps.begin(), timestamps.end(),
        [](const std::string &a, const std::string &b) {
            return std::stoull(a) < std::stoull(b);
        });

    for (const auto &timestamp : timestamps) {
        vstrImageFilenamesRGB.push_back(rgbByTimestamp.at(timestamp));
        vstrImageFilenamesD.push_back(depthByTimestamp.at(timestamp));
    }
    std::cout << "[MMPR] Found " << timestamps.size()
              << " timestamp-matched RGB-D frames." << std::endl;
}

// Read an uncompressed 2D NumPy array stored as little-endian float32.
// Python's np.save() normally produces this format for float32 depth maps.
// Returns CV_32FC1 with depth values UNCHANGED (meters in this dataset).
// Supports NPY v1/v2/v3 without requiring a third-party .npy library.
cv::Mat LoadDepthNPY(const std::filesystem::path &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open MMPR depth NPY: " + filename.string());

    char prefix[8];
    if (!file.read(prefix, sizeof(prefix)) ||
        std::memcmp(prefix, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a valid NPY file: " + filename.string());
    }

    const auto major = static_cast<unsigned char>(prefix[6]);
    std::uint32_t headerLength = 0;
    if (major == 1) {
        unsigned char length[2];
        if (!file.read(reinterpret_cast<char *>(length), 2))
            throw std::runtime_error("Truncated NPY header length");
        headerLength = static_cast<std::uint32_t>(length[0]) |
                       (static_cast<std::uint32_t>(length[1]) << 8);
    } else if (major == 2 || major == 3) {
        unsigned char length[4];
        if (!file.read(reinterpret_cast<char *>(length), 4))
            throw std::runtime_error("Truncated NPY header length");
        for (int i = 0; i < 4; ++i)
            headerLength |= static_cast<std::uint32_t>(length[i]) << (8 * i);
    } else {
        throw std::runtime_error("Unsupported NPY version in: " + filename.string());
    }

    if (headerLength == 0 || headerLength > (1u << 20))
        throw std::runtime_error("Invalid NPY header length: " + filename.string());
    std::string header(headerLength, '\0');
    if (!file.read(header.data(), header.size()))
        throw std::runtime_error("Truncated NPY header: " + filename.string());

    const std::regex dtypePattern(R"(['"]descr['"]\s*:\s*['"]([^'"]+)['"])");
    const std::regex fortranPattern(R"(['"]fortran_order['"]\s*:\s*(True|False))");
    const std::regex shapePattern(R"(['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\))");
    std::smatch dtypeMatch, fortranMatch, shapeMatch;
    if (!std::regex_search(header, dtypeMatch, dtypePattern) ||
        !std::regex_search(header, fortranMatch, fortranPattern) ||
        !std::regex_search(header, shapeMatch, shapePattern)) {
        throw std::runtime_error("Unsupported NPY header (need 2D float32): " + filename.string());
    }

    const std::string dtype = dtypeMatch[1].str();
    if (dtype != "<f4" && dtype != "=f4" && dtype != "|f4")
        throw std::runtime_error("MMPR depth NPY must contain little-endian float32, got: " + dtype);
    if (fortranMatch[1].str() != "False")
        throw std::runtime_error("Fortran-ordered NPY depth is not supported: " + filename.string());

    const auto rows = std::stoull(shapeMatch[1].str());
    const auto cols = std::stoull(shapeMatch[2].str());
    if (!rows || !cols || rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        rows * cols > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
        throw std::runtime_error("Invalid MMPR depth array size: " + filename.string());
    }

    cv::Mat depth(static_cast<int>(rows), static_cast<int>(cols), CV_32FC1);
    const auto bytes = static_cast<std::streamsize>(rows * cols * sizeof(float));
    if (!file.read(reinterpret_cast<char *>(depth.data), bytes))
        throw std::runtime_error("Truncated NPY depth data: " + filename.string());
    return depth;
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath) {
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    // std::sort(vTimesTrack.begin(), vTimesTrack.end());
    // out << "-------" << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "mean tracking time: " << totaltime / nImages << std::endl;

    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave) {
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10::CachingAllocator::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10::CachingAllocator::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10::CachingAllocator::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10::CachingAllocator::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}
