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

#include "image_wave.h"

#define SQR(x) ((x)*(x))

int image_wave_grid_init(struct ImageWaveGrid *grid,
                         int grid_w, int grid_h,
                         int image_w, int image_h)
{
    if (grid->array != NULL)
        free(grid->array);

    grid->grid_w = grid_w;
    grid->grid_h = grid_h;
    grid->image_w = image_w;
    grid->image_h = image_h;
    grid->array = calloc(grid->grid_w * grid->grid_h * 2, sizeof(double));
    if (grid->array == NULL)
        return -1;
    return 0;
}

void image_wave_grid_finalize(struct ImageWaveGrid *grid)
{
    if (grid->array != NULL)
    {
        free(grid->array);
        grid->array = NULL;
    }
}

void image_wave_grid_print(const struct ImageWaveGrid *grid)
{
    int i, j, k;
    for (i = 0; i < grid->grid_h; i++)
    {
        for (j = 0; j < grid->grid_w; j++)
        {
            printf("[");
            for (k = 0; k < 2; k++)
                printf("%lf ", image_wave_get_array(grid, j, i, k));
            printf("]");
        }
        printf("\n");
    }
    printf("\n");
}

void image_wave_grid_constant_shift(struct ImageWaveGrid *grid, double dx, double dy)
{
    int xi, yi;
    for (yi = 0; yi < grid->grid_h; yi++)
    {
        for (xi = 0; xi < grid->grid_w; xi++)
        {
            image_wave_set_array(grid, xi, yi, 0, dx);
            image_wave_set_array(grid, xi, yi, 1, dy);
        }
    }
}
