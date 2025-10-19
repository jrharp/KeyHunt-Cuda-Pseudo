/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdint.h>

// ---------------------------------------------------------------------------------
// Base58
// ---------------------------------------------------------------------------------

__device__ __constant__ char pszBase58[58] = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

__device__ __constant__ int8_t b58digits_map[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1, -1, -1, -1, -1, -1,
    -1,  9, 10, 11, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, -1, -1, -1, -1, -1,
    -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, -1, -1, -1, -1, -1};

__device__ __noinline__ void _GetAddress(uint32_t *__restrict__ hash, char *__restrict__ b58Add)
{

    uint32_t addBytes[16];
    uint32_t s[16];
    unsigned char A[25];
    unsigned char *addPtr = A;
    int retPos = 0;
    unsigned char digits[128];

    A[0] = 0x00;
    const unsigned char *hashBytes = reinterpret_cast<const unsigned char *>(hash);
    for (int i = 0; i < 20; i++)
        A[1 + i] = hashBytes[i];

    // Compute checksum

    addBytes[0] = __byte_perm(hash[0], static_cast<uint32_t>(A[0]), 0x4012);
    addBytes[1] = __byte_perm(hash[0], hash[1], 0x3456);
    addBytes[2] = __byte_perm(hash[1], hash[2], 0x3456);
    addBytes[3] = __byte_perm(hash[2], hash[3], 0x3456);
    addBytes[4] = __byte_perm(hash[3], hash[4], 0x3456);
    addBytes[5] = __byte_perm(hash[4], 0x80u, 0x3456);
    for (int i = 6; i < 15; i++)
        addBytes[i] = 0u;
    addBytes[15] = 0xA8u;

    SHA256Initialize(s);
    SHA256Transform(s, addBytes);

#pragma unroll 8
    for (int i = 0; i < 8; i++)
        addBytes[i] = s[i];

    addBytes[8] = 0x80000000;
    for (int i = 9; i < 15; i++)
        addBytes[i] = 0u;
    addBytes[15] = 0x100u;

    SHA256Initialize(s);
    SHA256Transform(s, addBytes);

    const unsigned char *stateBytes = reinterpret_cast<const unsigned char *>(s);
    A[21] = stateBytes[3];
    A[22] = stateBytes[2];
    A[23] = stateBytes[1];
    A[24] = stateBytes[0];

    // Base58

    // Skip leading zeroes
    while (addPtr[0] == 0) {
        b58Add[retPos++] = '1';
        addPtr++;
    }
    int length = 25 - retPos;

    int digitslen = 1;
    digits[0] = 0;
    for (int i = 0; i < length; i++) {
        uint32_t carry = addPtr[i];
        for (int j = 0; j < digitslen; j++) {
            carry += (uint32_t)(digits[j]) << 8;
            digits[j] = (unsigned char)(carry % 58);
            carry /= 58;
        }
        while (carry > 0) {
            digits[digitslen++] = (unsigned char)(carry % 58);
            carry /= 58;
        }
    }

    // reverse
    for (int i = 0; i < digitslen; i++)
        b58Add[retPos++] = (pszBase58[digits[digitslen - 1 - i]]);

    b58Add[retPos] = 0;

}
