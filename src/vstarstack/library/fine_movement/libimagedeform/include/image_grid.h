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
 * \brief Image data
 */
struct ImageGrid
{
    /**
     * @brief Image width
     */
    int w;
    /**
     * @brief Image height
     */
    int h;
    /**
     * @brief image data
     * Pixelds data, ordered line by line. Each line has length = image width. 
     */
    double *array;
};

/**
 * @brief Init image grid with 0
 * 
 * @param image image structure
 * @param width image width
 * @param height image height
 * @return int 0 for OK
 */
int image_grid_init(struct ImageGrid *image, int width, int height);

/**
 * @brief Free image grid
 * 
 * @param image image structure
 */
void image_grid_finalize(struct ImageGrid *image);

/**
 * \brief Set pixel at pos (x,y)
 * \param image Image array
 * \param x x
 * \param y y
 * \param val pixel value
 */
static inline void image_grid_set_pixel(struct ImageGrid *image,
                                        int x, int y, double val)
{
    if (x < 0 || y < 0 || x >= image->w || y >= image->h)
        return;
    image->array[y * image->w + x] = val;
}

/**
 * \brief Get pixel at pos (x,y)
 * \param image Image array
 * \param x x
 * \param y y
 * \return pixel value
 */
double image_grid_get_pixel(const struct ImageGrid *image,
                            double x, double y);

/**
 * \brief Global correlation between 2 images
 * \param image1 first image
 * \param image2 second image
 * \return correlator
 */
double image_grid_correlation(const struct ImageGrid *image1,
                              const struct ImageGrid *image2);

/**
 * \brief Get area of image 'img' and place it to 'area'
 * \param img Source image
 * \param x x position of area center
 * \param y y position of area center
 * \param area allocated ImageGrid for area content with specified w and h
 */
void image_grid_get_area(const struct ImageGrid *img,
                         double x, double y,
                         struct ImageGrid *area);
