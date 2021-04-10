__author__ = 'jieqing jiao'
__email__ = "jieqing.jiao@gmail.com"

from math import cos, sin

import numpy as np
import numpy.matlib
from numpy.linalg import solve
from scipy.interpolate import interpn


def rigid_p2matrix(translation, rotation):
    '''
    :param translation: shape (3,)
    :param rotation: shape (3,)
    :return T: shape(4,4) translation matrix
    :return R: shape(4,4) rotation matrix
    '''
    t_matrix = np.eye(4)
    r_x_matrix = np.eye(4)
    r_y_matrix = np.eye(4)
    r_z_matrix = np.eye(4)
    t_matrix[:3, -1] = translation
    phi = rotation[0]
    r_x_matrix[1, 1] = cos(phi)
    r_x_matrix[1, 2] = sin(phi)
    r_x_matrix[2, 1] = -sin(phi)
    r_x_matrix[2, 2] = cos(phi)
    phi = rotation[1]
    r_y_matrix[0, 0] = cos(phi)
    r_y_matrix[0, 2] = sin(phi)
    r_y_matrix[2, 0] = -sin(phi)
    r_y_matrix[2, 2] = cos(phi)
    phi = rotation[2]
    r_z_matrix[0, 0] = cos(phi)
    r_z_matrix[0, 1] = sin(phi)
    r_z_matrix[1, 0] = -sin(phi)
    r_z_matrix[1, 1] = cos(phi)
    r_matrix = r_x_matrix @ r_y_matrix @ r_z_matrix
    return t_matrix, r_matrix


def affine_transform_centre(xv, yv, zv, translation, rotation, centre, order='T*Tc_i*R*Tc'):
    '''
    :param xv: mesh grid
    :param yv: mesh grid
    :param zv: mesh grid
    :param translation:
    :param rotation:
    :param centre: centre for rotation
    :param order: 'Tc_i*R*Tc*T' or 'T*Tc_i*R*Tc'
    :return:
    '''
    t_matrix, r_matrix = rigid_p2matrix(translation, rotation)
    t_c_i_matrix, _ = rigid_p2matrix(centre, [0, 0, 0])
    t_c_matrix, _ = rigid_p2matrix([-c for c in centre], [0, 0, 0])
    m = np.eye(4)
    if order == 'T*Tc_i*R*Tc':
        m = t_matrix @ t_c_i_matrix @ r_matrix @ t_c_matrix
    if order == 'Tc_i*R*Tc*T':
        m = t_c_i_matrix @ r_matrix @ t_c_matrix @ t_matrix

    coordinates = np.row_stack((xv.flatten(), yv.flatten(), zv.flatten(), np.ones(xv.size)))
    coordinates_new = m @ coordinates
    xv_new = coordinates_new[0, :].reshape(xv.shape)
    yv_new = coordinates_new[1, :].reshape(yv.shape)
    zv_new = coordinates_new[2, :].reshape(zv.shape)
    return xv_new, yv_new, zv_new


def d_coordinates_over_d_transform(xv, yv, zv, rotation, centre):
    xv = xv.flatten()
    yv = yv.flatten()
    zv = zv.flatten()
    p4, p5, p6 = rotation[:3]
    c1, c2, c3 = centre[:3]
    ones = np.ones(xv.size)

    dx_over_dp = np.column_stack(
        (ones * 1, ones * 0, ones * 0, ones * 0,
         cos(p5) * zv - c3 * cos(p5) - yv * sin(p5) * sin(p6) + c1 * cos(p6) * sin(p5) +
         c2 * sin(p5) * sin(p6) - cos(p6) * xv * sin(p5), c1 * cos(p5) * sin(p6) -
         c2 * cos(p5) * cos(p6) + cos(p5) * cos(p6) * yv - cos(p5) * xv * sin(p6)))

    dy_over_dp = np.column_stack(
        (ones * 0, ones * 1, ones * 0, c2 * (cos(p6) * sin(p4) + cos(p4) * sin(p5) * sin(p6)) -
         c1 * (sin(p4) * sin(p6) - cos(p4) * cos(p6) * sin(p5)) + xv *
         (sin(p4) * sin(p6) - cos(p4) * cos(p6) * sin(p5)) - yv *
         (cos(p6) * sin(p4) + cos(p4) * sin(p5) * sin(p6)) - c3 * cos(p4) * cos(p5) +
         cos(p4) * cos(p5) * zv, c3 * sin(p4) * sin(p5) - zv * sin(p4) * sin(p5) +
         c1 * cos(p5) * cos(p6) * sin(p4) + c2 * cos(p5) * sin(p4) * sin(p6) -
         cos(p5) * cos(p6) * xv * sin(p4) - cos(p5) * yv * sin(p4) * sin(p6),
         c1 * (cos(p4) * cos(p6) - sin(p4) * sin(p5) * sin(p6)) + c2 *
         (cos(p4) * sin(p6) + cos(p6) * sin(p4) * sin(p5)) - xv *
         (cos(p4) * cos(p6) - sin(p4) * sin(p5) * sin(p6)) - yv *
         (cos(p4) * sin(p6) + cos(p6) * sin(p4) * sin(p5))))

    dz_over_dp = np.column_stack(
        (ones * 0, ones * 0, ones * 1, c2 * (cos(p4) * cos(p6) - sin(p4) * sin(p5) * sin(p6)) -
         c1 * (cos(p4) * sin(p6) + cos(p6) * sin(p4) * sin(p5)) + xv *
         (cos(p4) * sin(p6) + cos(p6) * sin(p4) * sin(p5)) - yv *
         (cos(p4) * cos(p6) - sin(p4) * sin(p5) * sin(p6)) + c3 * cos(p5) * sin(p4) -
         cos(p5) * zv * sin(p4), c3 * cos(p4) * sin(p5) - cos(p4) * zv * sin(p5) +
         c1 * cos(p4) * cos(p5) * cos(p6) + c2 * cos(p4) * cos(p5) * sin(p6) -
         cos(p4) * cos(p5) * cos(p6) * xv - cos(p4) * cos(p5) * yv * sin(p6),
         xv * (cos(p6) * sin(p4) + cos(p4) * sin(p5) * sin(p6)) - c2 *
         (sin(p4) * sin(p6) - cos(p4) * cos(p6) * sin(p5)) - c1 *
         (cos(p6) * sin(p4) + cos(p4) * sin(p5) * sin(p6)) + yv *
         (sin(p4) * sin(p6) - cos(p4) * cos(p6) * sin(p5))))

    return dx_over_dp, dy_over_dp, dz_over_dp


def image_rigid_transform_3d(image, translation, rotation):
    x_range = np.arange(image.shape[0])
    y_range = np.arange(image.shape[1])
    z_range = np.arange(image.shape[2])
    xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    centre = [xv.mean(), yv.mean(), zv.mean()]
    xv_new, yv_new, zv_new = affine_transform_centre(xv, yv, zv, translation, rotation, centre)
    image_new = interpn((x_range, y_range, z_range), image,
                        np.vstack(
                            (xv_new.flatten(), yv_new.flatten(), zv_new.flatten())).transpose(),
                        bounds_error=False)
    image_new = image_new.reshape(image.shape)
    return image_new


def image_registration_3d_gn(image, image_ref, translation, rotation, n_iter=100):
    '''
    rigid registration of two 3d images using Gauss-Newton method
    :param image:
    :param image_ref:
    :param translation:
    :param rotation:
    :param n_iter:
    :return:
    '''
    x_range = np.arange(image.shape[0])
    y_range = np.arange(image.shape[1])
    z_range = np.arange(image.shape[2])
    xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    centre = [xv.mean(), yv.mean(), zv.mean()]
    for i in range(n_iter):
        print(i)
        xv_new, yv_new, zv_new = affine_transform_centre(xv, yv, zv, translation, rotation, centre)
        image_new = interpn(
            (x_range, y_range, z_range), image,
            np.vstack((xv_new.flatten(), yv_new.flatten(), zv_new.flatten())).transpose(),
            bounds_error=False)
        image_new = image_new.reshape(image.shape)
        dx_over_dp, dy_over_dp, dz_over_dp = d_coordinates_over_d_transform(
            xv, yv, zv, rotation, centre)
        image_new_g = image_new * 1
        image_new_g[np.isnan(image_new)] = 0
        image_gradient = np.gradient(image_new_g)
        image_gradient_x = np.matlib.repmat(image_gradient[0].flatten(), dx_over_dp.shape[-1],
                                            1).transpose()
        image_gradient_y = np.matlib.repmat(image_gradient[1].flatten(), dy_over_dp.shape[-1],
                                            1).transpose()
        image_gradient_z = np.matlib.repmat(image_gradient[2].flatten(), dz_over_dp.shape[-1],
                                            1).transpose()
        jacobian = (image_gradient_x*dx_over_dp + image_gradient_y*dy_over_dp +
                    image_gradient_z*dz_over_dp)
        image_diff = image_new - image_ref
        image_diff[np.isnan(image_diff)] = 0
        a = jacobian.transpose() @ jacobian
        update = solve(a, jacobian.transpose()) @ image_diff.flatten()
        translation = np.subtract(translation, update[:3])
        rotation = np.subtract(rotation, update[3:])
        print(translation)
        print(rotation)
        if all(abs(update) < 0.01):
            break
    return translation, rotation, image_new_g


def image_registration_3d_plus_t(image_t, image_ref_t, translation_t, rotation_t,
                                 registration_index, n_iter=100):
    image_new_t = image_t * 1
    for t in registration_index:
        translation_t[t], rotation_t[t], image_new_t[:, :, :, t] = \
            image_registration_3d_gn(
                image_t[:, :, :, t], image_ref_t[:, :, :, t],
                translation=translation_t[t], rotation=rotation_t[t], n_iter=n_iter)
    return translation_t, rotation_t, image_new_t
