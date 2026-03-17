#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief initializes a vector with values from 0 to length-1 multiplied by factor
 * @param vec the vector to initialize
 * @param length the length of the vector
 * @param factor the factor to multiply each element by
 */
void initVec(float* vec, int length, int factor);

/**
 * @brief initializes a vector with values from a file
 * @param vec the vector to initialize
 * @param length the length of the vector
 * @param filePath the path to the file containing the values
 */
void initVecFile(float* vec, int length,const std::string& filePath);

/**
 * @brief creates a file containing 32-bit floating-point values
 * @param filePath the path to the file to create
 * @param length the number of values to generate
 * @param factor the factor to multiply each element by
 */
void create32File(const std::string& filePath , int length, int factor);

/**
 * @brief writes the content of a vector to a file
 * @param filePath the path to the file to create
 * @param vec the vector to write
 * @param length the length of the vector
 */
void ansToFile(const std::string& filePath, float* vec, int length);

/**
 * @brief reads the content of a file and prints it to the console
 * @param filePath the path to the file to read
 * @param length the number of values to read from the file
 */
void printFileContent(const std::string& filePath, int length);

