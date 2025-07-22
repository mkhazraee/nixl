/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <gtest/gtest.h>
#include "utils/file/file_utils.h"

class FileUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test files
        test_file1 = "/tmp/nixl_test_file_1.txt";
        test_file2 = "/tmp/nixl_test_file_2.txt";
        non_existent_file = "/tmp/nixl_test_nonexistent_file.txt";

        // Create test file 1
        {
            std::ofstream test_file(test_file1);
            test_file << "Test content for file 1" << std::endl;
            test_file.close();
        }

        // Create test file 2
        {
            std::ofstream test_file(test_file2);
            test_file << "Test content for file 2" << std::endl;
            test_file.close();
        }
    }

    void TearDown() override {
        // Clean up
        std::filesystem::remove(test_file1);
        std::filesystem::remove(test_file2);
    }

    std::string test_file1;
    std::string test_file2;
    std::string non_existent_file;
};

TEST_F(FileUtilsTest, QueryFileInfoWithExistingFile) {
    auto result = nixl::queryFileInfo(test_file1);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->find("size") != result->end());
    EXPECT_TRUE(result->find("mode") != result->end());
    EXPECT_TRUE(result->find("mtime") != result->end());
}

TEST_F(FileUtilsTest, QueryFileInfoWithNonExistentFile) {
    auto result = nixl::queryFileInfo(non_existent_file);

    EXPECT_FALSE(result.has_value());
}

TEST_F(FileUtilsTest, QueryFileInfoWithEmptyFilename) {
    auto result = nixl::queryFileInfo("");

    EXPECT_FALSE(result.has_value());
}

TEST_F(FileUtilsTest, QueryFileInfoListWithMultipleExistingFiles) {
    std::vector<std::string> filenames = {test_file1, test_file2};
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 2);
    EXPECT_TRUE(resp[0].accessible);
    EXPECT_TRUE(resp[1].accessible);
    EXPECT_TRUE(resp[0].info.find("size") != resp[0].info.end());
    EXPECT_TRUE(resp[1].info.find("size") != resp[1].info.end());
}

TEST_F(FileUtilsTest, QueryFileInfoListWithMixedFiles) {
    std::vector<std::string> filenames = {test_file1, non_existent_file, test_file2};
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 3);
    EXPECT_TRUE(resp[0].accessible);   // test_file1 exists
    EXPECT_FALSE(resp[1].accessible);  // non_existent_file doesn't exist
    EXPECT_TRUE(resp[2].accessible);   // test_file2 exists
    EXPECT_TRUE(resp[0].info.find("size") != resp[0].info.end());
    EXPECT_TRUE(resp[1].info.empty()); // No info for non-existent file
    EXPECT_TRUE(resp[2].info.find("size") != resp[2].info.end());
}

TEST_F(FileUtilsTest, QueryFileInfoListWithEmptyVector) {
    std::vector<std::string> filenames;
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_TRUE(resp.empty());
}

TEST_F(FileUtilsTest, QueryFileInfoListWithEmptyFilenames) {
    std::vector<std::string> filenames = {"", "", ""};
    std::vector<nixl_query_resp_t> resp;
    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 3);
    EXPECT_FALSE(resp[0].accessible);
    EXPECT_FALSE(resp[1].accessible);
    EXPECT_FALSE(resp[2].accessible);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
