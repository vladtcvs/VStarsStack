/*
 * Copyright (c) 2023-2024 Vladislav Tsendrovskii
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief 1-Dim cubic interpolation for x in range [0..1]
 * \param fm1 function value at x=-1, can be NAN
 * \param f0 function value at x=0
 * \param f1 function value at x=1
 * \param f2 function value at x=2, can be NAN
 * \param x x position
 * \return interpolated value
 */
real_t interpolation_1d_cubic(real_t fm1, real_t f0, real_t f1, real_t f2, real_t x);

/**
 * \brief 2-Dim cubic interpolation for x and y in range [0..1]
 * \param x x position
 * \param y y position
 * \return interpolated value
 */
real_t interpolation_2d_cubic(real_t fm1m1, real_t f0m1, real_t f1m1, real_t f2m1,
                              real_t fm10,  real_t f00,  real_t f10,  real_t f20,
                              real_t fm11,  real_t f01,  real_t f11,  real_t f21,
                              real_t fm12,  real_t f02,  real_t f12,  real_t f22,
                              real_t x, real_t y);

/**
 * \brief 1-Dim linear interpolation for x in range [0..1]
 * \param f0 function value at x=0
 * \param f1 function value at x=1
 * \param x x position
 * \return interpolated value
 */
real_t interpolation_1d_linear(real_t f0, real_t f1, real_t x);

/**
 * \brief 2-Dim linear interpolation for x and y in range [0..1]
 * \param x x position
 * \param y y position
 * \return interpolated value
 */
real_t interpolation_2d_linear(real_t f00,  real_t f10,
                               real_t f01,  real_t f11,
                               real_t x, real_t y);

#ifdef __cplusplus
}
#endif
