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

/**
 * \brief 1-Dim interpolation for x in range [0..1]
 * \param fm1 function value at x=-1, can be NAN
 * \param f0 function value at x=0
 * \param f1 function value at x=1
 * \param f2 function value at x=2, can be NAN
 * \return interpolated value
 */
double interpolation_1d(double fm1, double f0, double f1, double f2, double x);

/**
 * \brief 2-Dim interpolation for x and y in range [0..1]
 * \return interpolated value
 */
double interpolation_2d(double fm1m1, double f0m1, double f1m1, double f2m1,
                        double fm10,  double f00,  double f10,  double f20,
                        double fm11,  double f01,  double f11,  double f21,
                        double fm12,  double f02,  double f12,  double f22,
                        double x, double y);
