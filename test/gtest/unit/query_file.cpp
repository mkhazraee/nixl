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

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <optional>
#include "file/file_utils.h"
#include "nixl.h"
#include "nixl_descriptors.h"

class QueryFileTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // Create temporary test files
        test_file1 = "/tmp/test_query_mem_1.txt";
        test_file2 = "/tmp/test_query_mem_2.txt";
        non_existent_file = "/tmp/non_existent_file.txt";

        // Create test file 1
        {
            std::ofstream file(test_file1);
            file << "Test content for file 1" << std::endl;
            file.close();
        }

        // Create test file 2
        {
            std::ofstream file(test_file2);
            file << "Test content for file 2" << std::endl;
            file.close();
        }
    }

    void
    TearDown() override {
        // Clean up test files
        std::filesystem::remove(test_file1);
        std::filesystem::remove(test_file2);
    }

    std::string test_file1;
    std::string test_file2;
    std::string non_existent_file;
};

TEST_F(QueryFileTest, QueryFileInfoListWithEmptyVector) {
    std::vector<std::string> filenames;
    std::vector<nixl_query_resp_t> resp;

    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 0);
}

TEST_F(QueryFileTest, QueryFileInfoListWithEmptyFilenames) {
    std::vector<std::string> filenames = {"", "", ""};
    std::vector<nixl_query_resp_t> resp;

    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 3);
    EXPECT_FALSE(resp[0].has_value());
    EXPECT_FALSE(resp[1].has_value());
    EXPECT_FALSE(resp[2].has_value());
}

TEST_F(QueryFileTest, QueryFileInfoListWithMultipleExistingFiles) {
    std::vector<std::string> filenames = {test_file1, test_file2};
    std::vector<nixl_query_resp_t> resp;

    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 2);
    EXPECT_TRUE(resp[0].has_value());
    EXPECT_TRUE(resp[1].has_value());
    EXPECT_TRUE(resp[0].value().find("size") != resp[0].value().end());
    EXPECT_TRUE(resp[1].value().find("size") != resp[1].value().end());
}

TEST_F(QueryFileTest, QueryFileInfoListWithMixedFiles) {
    std::vector<std::string> filenames = {test_file1, non_existent_file, test_file2, ""};
    std::vector<nixl_query_resp_t> resp;

    nixl_status_t status = nixl::queryFileInfoList(filenames, resp);
    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_EQ(resp.size(), 4);
    EXPECT_TRUE(resp[0].has_value()); // test_file1 exists
    EXPECT_FALSE(resp[1].has_value()); // non_existent_file doesn't exist
    EXPECT_TRUE(resp[2].has_value()); // test_file2 exists
    EXPECT_FALSE(resp[3].has_value()); // empty filename doesn't exist
}
