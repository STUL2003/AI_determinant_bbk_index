import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def gauss_jordan(matrix):
    n = len(matrix)
    args = [row + [int(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = args[i][i]
        for j in range(2 * n):
            args[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = args[k][i]
                for j in range(2 * n):
                    args[k][j] -= factor * args[i][j]
    matrix_inv = [row[n:] for row in args]
    return matrix_inv


# Массобаритные характеристики аппарата I - моменты инерции
I_x = 0.0022
I_y = 0.0029
I_z = 0.0048
m = 0.429
l = 0.178  # размах плеча

g = 9.81  # ускорение свободного падения

# Вспомогательные параметры
a1 = (I_y - I_z) / I_x
a3 = (I_z - I_x) / I_y
a5 = (I_x - I_y) / I_z
b1 = l / I_x
b2 = l / I_y
b3 = l / I_z

dt = 0.5  # шаг моделирования (уменьшен для лучшей численной устойчивости)
Ln = 1000  # число шагов моделирования

# Начальные условия - координаты аппарата
x = [0] * Ln
y = [0] * Ln
z = [0] * Ln

# Начальные условия - скорости аппарата (в наземной системе координат)
x_d = [0] * Ln
y_d = [0] * Ln
z_d = [0] * Ln

# Начальные условия - углы ориентации
phi = [0] * Ln  # крен
theta = [0] * Ln  # тангаж
psi = [0] * Ln  # рыскание

# Начальные условия - скорости изменения углов ориентации
phi_d = [0] * Ln
theta_d = [0] * Ln
psi_d = [0] * Ln

psid = [0] * Ln
phid = [0] * Ln
thetad = [0] * Ln

# Константы регуляторов
kp_ang = 0.01
kd_ang = 0.01
ki_ang = 0.0001

alpha = 0.1  # коэффициент сглаживания
Vc = 0.5  # скорости цели

xc = np.zeros(Ln)
yc = np.zeros(Ln)
zc = np.zeros(Ln)

xc[0] = 100
yc[0] = -100
zc[0] = 60

acceleration = 0.1  # Ускорение
deceleration = 0.1  # Замедление
max_speed = 1  # Максимальная скорость

vx = 0
vy = 0
vz = 0

num_targets = 5
target_points = [(random.uniform(-200, 200), random.uniform(-200, 200), random.uniform(0, 100)) for _ in
                 range(num_targets)]

current_target_index = 0
target_x, target_y, target_z = target_points[current_target_index]

for i in range(1, Ln):
    if i % 200 == 0:  # меняем цель каждые 200 шагов
        current_target_index = (current_target_index + 1) % num_targets
        target_x, target_y, target_z = target_points[current_target_index]

    # расчет расстояния до цели
    dx = target_x - xc[i - 1]
    dy = target_y - yc[i - 1]
    dz = target_z - zc[i - 1]

    # расчет направления движения
    direction_x = dx / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    direction_y = dy / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    direction_z = dz / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # ускорение или замедление в зависимости от расстояния до цели
    if np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) > 10:
        vx += acceleration * direction_x
        vy += acceleration * direction_y
        vz += acceleration * direction_z
    else:
        vx -= deceleration * direction_x
        vy -= deceleration * direction_y
        vz -= deceleration * direction_z

    # ограничение скорости
    vx = np.clip(vx, -max_speed, max_speed)
    vy = np.clip(vy, -max_speed, max_speed)
    vz = np.clip(vz, -max_speed, max_speed)

    # обновление позиции
    xc[i] = xc[i - 1] + vx * dt
    yc[i] = yc[i - 1] + vy * dt
    zc[i] = zc[i - 1] + vz * dt

for i in range(1, Ln):
    xc[i] = alpha * xc[i] + (1 - alpha) * xc[i - 1]
    yc[i] = alpha * yc[i] + (1 - alpha) * yc[i - 1]
    zc[i] = alpha * zc[i] + (1 - alpha) * zc[i - 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-200, 200])
ax.view_init(60, 60)

phis = 0
thetas = 0
psis = 0

b = False
j = 2

# PID control variables
integral_error_x = 0
integral_error_y = 0
integral_error_z = 0
previous_error_x = 0
previous_error_y = 0
previous_error_z = 0

# Define frustum parameters
fov_horizontal = 60  # Field of view horizontal angle in degrees
fov_vertical = 45  # Field of view vertical angle in degrees
near_plane = 1  # Distance to near plane
far_plane = 100  # Distance to far plane

# Convert FOV to radians
fov_horizontal_rad = math.radians(fov_horizontal)
fov_vertical_rad = math.radians(fov_vertical)

# Define proximity threshold as a percentage of FOV
proximity_threshold = 0.8  # 80% of FOV


# Function to compute frustum vertices
def compute_frustum(phi, theta, psi, position, near, far):
    # Convert angles to radians
    phi_rad = phi
    theta_rad = theta
    psi_rad = psi

    # Compute rotation matrix
    R = rotation_matrix(phi_rad, theta_rad, psi_rad)

    # Define frustum points in camera coordinates
    aspect = math.tan(math.radians(fov_horizontal / 2)) / math.tan(math.radians(fov_vertical / 2))
    h_near = near * math.tan(math.radians(fov_vertical / 2))
    w_near = h_near * aspect
    h_far = far * math.tan(math.radians(fov_vertical / 2))
    w_far = h_far * aspect

    # Define near and far plane vertices
    near_points = [
        np.array([near, -w_near, -h_near]),
        np.array([near, w_near, -h_near]),
        np.array([near, w_near, h_near]),
        np.array([near, -w_near, h_near])
    ]

    far_points = [
        np.array([far, -w_far, -h_far]),
        np.array([far, w_far, -h_far]),
        np.array([far, w_far, h_far]),
        np.array([far, -w_far, h_far])
    ]

    # Transform to global coordinates
    global_near_points = [R @ p + position for p in near_points]
    global_far_points = [R @ p + position for p in far_points]

    # Create frustum faces
    faces = [
        [global_near_points[0], global_far_points[0], global_far_points[1], global_near_points[1]],
        [global_near_points[1], global_far_points[1], global_far_points[2], global_near_points[2]],
        [global_near_points[2], global_far_points[2], global_far_points[3], global_near_points[3]],
        [global_near_points[3], global_far_points[3], global_far_points[0], global_near_points[0]],
        [global_near_points[0], global_near_points[1], global_near_points[2], global_near_points[3]],
        [global_far_points[0], global_far_points[3], global_far_points[2], global_far_points[1]]
    ]

    return faces


# PID gains for visual servoing
kp_visual = 0.1
kd_visual = 0.05
ki_visual = 0.001

# Initialize PID variables for visual servoing
integral_error_visual = 0
previous_error_visual = 0

# Initialize arrays to store data for plotting
distance_data = []
time_to_approach_data = []
speed_data = []
offset_x_data = []
offset_y_data = []


def rotation_matrix(phi, theta, psi):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(phi), -math.sin(phi)],
                    [0, math.sin(phi), math.cos(phi)]])

    R_y = np.array([[math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])

    R_z = np.array([[math.cos(psi), -math.sin(psi), 0],
                    [math.sin(psi), math.cos(psi), 0],
                    [0, 0, 1]])

    R = R_z @ R_y @ R_x
    return R


for i in range(2, Ln):
    r = math.sqrt((xc[i] - x[i - 1]) ** 2 + (yc[i] - y[i - 1]) ** 2 + (zc[i] - z[i - 1]) ** 2)
    speed_to_target = r / dt

    # Законы управления по углам ориентации
    Uxd = kp_ang * (xc[i] - x[i - 1]) + kd_ang * ((xc[i] - 2 * xc[i - 1] + xc[i - 2]) / dt ** 2 - x_d[i - 1])
    Uyd = kp_ang * (yc[i] - y[i - 1]) + kd_ang * ((yc[i] - 2 * yc[i - 1] + yc[i - 2]) / dt ** 2 - y_d[i - 1])
    Uzd = kp_ang * (zc[i] - z[i - 1]) / (math.cos(phi[i - 1]) * math.cos(theta[i - 1])) + kd_ang * (
                (zc[i] - 2 * zc[i - 1] + zc[i - 2]) / dt ** 2 - z_d[i - 1]) + m * g

    U1 = Uzd
    U0 = math.sqrt(Uxd ** 2 + Uyd ** 2 + Uzd ** 2)

    # Требуемые углы ориентации
    psid[i - 1] = math.atan2(Uyd, Uxd)

    dx = xc[i] - x[i]
    dy = yc[i] - y[i]
    dz = zc[i] - z[i]

    offset_x = dy * math.cos(psi[i]) - dx * math.sin(psi[i])
    offset_y = dz * math.cos(theta[i]) - dx * math.sin(theta[i])

    offset_x_data.append(offset_x)
    offset_y_data.append(offset_y)

    # Проверка на допустимый диапазон для math.asin
    arg_phi = (Uxd / Uzd * math.sin(psid[i - 1]) + Uyd / Uzd * math.cos(psid[i - 1]))
    arg_phi = np.clip(arg_phi, -1, 1)
    phid[i - 1] = math.asin(arg_phi) + math.atan(offset_y / r)

    arg_theta = (Uxd / Uzd * math.cos(psid[i - 1]) - Uyd / Uzd * math.sin(psid[i - 1]))
    arg_theta = max(min(arg_theta, 1), -1)
    thetad[i - 1] = math.asin(arg_theta)

    # Законы управления по углам ориентации
    phi_error = phid[i - 1] - phi[i - 1]
    theta_error = thetad[i - 1] - theta[i - 1]
    psi_error = psid[i - 1] - psi[i - 1]

    # Интегральная составляющая
    phis += phi_error * dt
    thetas += theta_error * dt
    psis += psi_error * dt

    U2 = kp_ang * phi_error + kd_ang * ((phid[i - 1] - phid[i - 2]) / dt - phi_d[i - 1]) + ki_ang * phis
    U3 = kp_ang * theta_error + kd_ang * ((thetad[i - 1] - thetad[i - 2]) / dt - theta_d[i - 1]) + ki_ang * thetas
    U4 = kp_ang * psi_error + kd_ang * ((psid[i - 1] - psid[i - 2]) / dt - psi_d[i - 1]) + ki_ang * psis

    # Расчет управляющих воздействий для двигателей
    matrix_inv = gauss_jordan([[1, 1, 1, 1], [0, -1, 0, 1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
    list_ = [U1, U2, U3, U4]
    V0 = [sum(matrix_inv[i][j] * list_[j] for j in range(4)) for i in range(4)]
    V0 = [max(0, v) for v in V0]
    matrix = [[1, 1, 1, 1], [0, -1, 0, 1], [-1, 0, 1, 0], [-1, 1, -1, 1]]
    UU = [sum(matrix[i][j] * V0[j] for j in range(4)) for i in range(4)]
    U1, U2, U3, U4 = UU

    # Расчет вторых производных (ускорений) для углов ориентации
    phi_dd = (l * U2 + (I_y - I_z) * psi_d[i - 1] * theta_d[i - 1]) / I_x
    theta_dd = (l * U3 + (I_x - I_z) * phi_d[i - 1] * psi_d[i - 1]) / I_y
    psi_dd = (l * U4 + (I_x - I_y) * phi_d[i - 1] * theta_d[i - 1]) / I_z

    # Расчет первых производных (скоростей изменения) для углов ориентации
    phi_d[i] = phi_d[i - 1] + phi_dd * dt
    theta_d[i] = theta_d[i - 1] + theta_dd * dt
    psi_d[i] = psi_d[i - 1] + psi_dd * dt

    # Ограничение угловых скоростей
    max_angular_rate = 0.5
    phi_d[i] = max(min(phi_d[i], max_angular_rate), -max_angular_rate)
    theta_d[i] = max(min(theta_d[i], max_angular_rate), -max_angular_rate)
    psi_d[i] = max(min(psi_d[i], max_angular_rate), -max_angular_rate)

    # Расчет углов ориентации
    phi[i] = phi[i - 1] + phi_d[i] * dt
    theta[i] = theta[i - 1] + theta_d[i] * dt
    psi[i] = psi[i - 1] + psi_d[i] * dt

    # Ограничение углов
    phi[i] = max(min(phi[i], 0.25), -0.25)
    theta[i] = max(min(theta[i], 0.25), -0.25)
    psi[i] = max(min(psi[i], math.pi), -math.pi)

    # Расчет вторых производных (ускорений) для координат
    x_dd = U1 / m * (math.cos(phi[i]) * math.sin(theta[i]) * math.cos(psi[i]) + math.sin(phi[i]) * math.sin(psi[i]))
    y_dd = -U1 / m * (math.cos(phi[i]) * math.sin(theta[i]) * math.sin(psi[i]) - math.cos(psi[i]) * math.sin(phi[i]))
    z_dd = U1 / m * (math.cos(theta[i]) * math.cos(phi[i])) - g

    # Расчет первых производных (скоростей аппарата) в наземной системе координат
    x_d[i] = x_d[i - 1] + x_dd * dt
    y_d[i] = y_d[i - 1] + y_dd * dt
    z_d[i] = z_d[i - 1] + z_dd * dt

    # Ограничение скорости дрона
    max_drone_speed = 5  # Максимальная скорость дрона
    x_d[i] = max(min(x_d[i], max_drone_speed), -max_drone_speed)
    y_d[i] = max(min(y_d[i], max_drone_speed), -max_drone_speed)
    z_d[i] = max(min(z_d[i], max_drone_speed), -max_drone_speed)

    # Ограничение скорости отдаления
    if x_d[i] > 0 and xc[i] < x[i - 1]:
        x_d[i] = max(x_d[i], 0)
    if y_d[i] > 0 and yc[i] < y[i - 1]:
        y_d[i] = max(y_d[i], 0)
    if z_d[i] > 0 and zc[i] < z[i - 1]:
        z_d[i] = max(z_d[i], 0)

    psi[i] = max(min(psi[i], math.pi), -math.pi)

    # Расчет координат
    x[i] = x[i - 1] + x_d[i] * dt
    y[i] = y[i - 1] + y_d[i] * dt
    z[i] = z[i - 1] + z_d[i] * dt

    j += 1

    if r < 5:
        print("Yes")
        print(i)
        b = True
        break

    # Calculate distance to the target
    distance = math.sqrt((x[i] - xc[i]) ** 2 + (y[i] - yc[i]) ** 2 + (z[i] - zc[i]) ** 2)
    distance_data.append(distance)

    # Calculate speed of the drone
    speed = math.sqrt(x_d[i] ** 2 + y_d[i] ** 2 + z_d[i] ** 2)
    speed_data.append(speed)

    # Avoid division by zero for time to approach
    if speed > 1e-6:
        time_to_approach = distance / speed
    else:
        time_to_approach = float('inf')  # or a large number
    time_to_approach_data.append(time_to_approach)

    # Calculate offset in the frame
    # Assume camera looks in the direction of the drone's forward vector
    # Calculate the forward vector based on orientation angles
    forward_x = math.cos(phi[i]) * math.sin(theta[i]) * math.cos(psi[i]) + math.sin(phi[i]) * math.sin(psi[i])
    forward_y = -math.cos(phi[i]) * math.sin(theta[i]) * math.sin(psi[i]) + math.cos(psi[i]) * math.sin(phi[i])
    forward_z = math.cos(theta[i]) * math.cos(phi[i])

    # Vector from drone to target
    dx = xc[i] - x[i]
    dy = yc[i] - y[i]
    dz = zc[i] - z[i]

    # Спроецируйте положение цели в систему координат камеры
    # Рассчитайте смещение в кадре камеры
    # Предполагая, что вектор движения камеры вверх основан на значениях phi и theta
    # Эта часть может потребовать корректировки в зависимости от конкретной модели камеры
    # Для простоты рассчитайте смещение в горизонтальной и вертикальной плоскостях
    # Смещение по горизонтали (ось X)
    offset_x = dy * math.cos(psi[i]) - dx * math.sin(psi[i])
    # Vertical offset (Y-axis)
    offset_y = dz * math.cos(theta[i]) - dx * math.sin(theta[i])

    offset_x_data.append(offset_x)
    offset_y_data.append(offset_y)

# Enable interactive mode
if b == True:
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='цель'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='дрон')
    ]
    local_forward = np.array([1, 0, 0])
    arrow_length = 20  # Increase this value to make the arrow longer
    arrow_length_ratio = 0.2  # Increase this value for a larger arrow head
    linewidth = 2
    ax.set_box_aspect([1, 1, 1])

    # Сохраняем данные для графиков вне цикла
    phi_data = phi
    theta_data = theta
    psi_data = psi
    x_data = x
    y_data = y
    z_data = z

    # Get the current figure
    fig = plt.gcf()

    # Initialize text annotation in figure coordinates
    text_annotation = fig.text(0.02, 0.95, '', transform=fig.transFigure, color='black', fontsize=10)

    for i in range(2, j):
        R = rotation_matrix(phi[i], theta[i], psi[i])
        global_forward = R @ local_forward

        # Plot the drone's position and the forward arrow
        drone_scatter = ax.scatter(x[i], y[i], z[i], c='r', marker='o', label="дрон")
        drone_quiver = ax.quiver(x[i], y[i], z[i],
                                 global_forward[0] * arrow_length,
                                 global_forward[1] * arrow_length,
                                 global_forward[2] * arrow_length,
                                 color='g', arrow_length_ratio=0.2, linewidth=2)

        # Plot the target position
        target_scatter = ax.scatter(xc[i], yc[i], zc[i], c='b', marker='o', label="цель")
        line1, = ax.plot([x[i], x[i] + x_d[i]], [y[i], y[i] + y_d[i]], [z[i], z[i] + z_d[i]], 'k-')
        line2, = ax.plot([x[i], xc[i]], [y[i], yc[i]], [z[i], zc[i]], 'k--')

        direction_x = math.cos(phi[i]) * math.sin(theta[i]) * math.cos(psi[i]) + math.sin(phi[i]) * math.sin(psi[i])
        direction_y = -math.cos(phi[i]) * math.sin(theta[i]) * math.sin(psi[i]) + math.cos(psi[i]) * math.sin(phi[i])
        direction_z = math.cos(theta[i]) * math.cos(phi[i])
        direction_quiver = ax.quiver(x[i], y[i], z[i], direction_x, direction_y, direction_z, color='g', length=10,
                                     lw=1)

        # Compute frustum based on current drone orientation and position
        frustum_faces = compute_frustum(phi[i], theta[i], psi[i], np.array([x[i], y[i], z[i]]), near_plane, far_plane)

        # Plot the frustum
        frustum_collections = []
        for face in frustum_faces:
            verts = [face[j] for j in range(4)]
            coll = Poly3DCollection([verts], facecolors='cyan', linewidths=1, edgecolors='b', alpha=0.2)
            frustum_collections.append(coll)
            ax.add_collection3d(coll)

        legend = ax.legend(handles=legend_elements, loc='upper right')

        # Calculate and display dynamic data
        distance = math.sqrt((x[i] - xc[i]) ** 2 + (y[i] - yc[i]) ** 2 + (z[i] - zc[i]) ** 2)
        speed = math.sqrt(x_d[i] ** 2 + y_d[i] ** 2 + z_d[i] ** 2)
        if speed > 1e-6:
            time_to_approach = distance / speed
        else:
            time_to_approach = float('inf')

        forward_x = math.cos(phi[i]) * math.sin(theta[i]) * math.cos(psi[i]) + math.sin(phi[i]) * math.sin(psi[i])
        forward_y = -math.cos(phi[i]) * math.sin(theta[i]) * math.sin(psi[i]) + math.cos(psi[i]) * math.sin(phi[i])
        forward_z = math.cos(theta[i]) * math.cos(phi[i])

        dx = xc[i] - x[i]
        dy = yc[i] - y[i]
        dz = zc[i] - z[i]

        offset_x = dy * math.cos(psi[i]) - dx * math.sin(psi[i])
        offset_y = dz * math.cos(theta[i]) - dx * math.sin(theta[i])

        # Update text annotation
        # Update text annotation
        # Set the text with the desired information
        text_annotation.set_text(
            f'Расстояние: {distance:.2f} м | Скорость: {speed:.2f} м/с | Время сближения: {time_to_approach:.2f} с \n\n Смещение X: {offset_x:.2f} м | Смещение Y: {offset_y:.2f} м')

        # Set the position of the annotation lower in the window
        text_annotation.set_position((0.1, 0.9))  # Adjust the y-value to move it up or down
        plt.pause(0.01)

        # Remove only the plot elements added in this iteration
        drone_scatter.remove()
        drone_quiver.remove()
        target_scatter.remove()
        line1.remove()
        line2.remove()
        direction_quiver.remove()
        for coll in frustum_collections:
            coll.remove()
        legend.remove()

        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim([-200, 200])
        ax.set_box_aspect([1, 1, 1])

    # После завершения цикла, строим графики
    plt.figure()

    plt.subplot(311)
    plt.plot(phi_data, label='Крен')  # Added label
    plt.ylabel('Крен')
    plt.legend()

    plt.subplot(312)
    plt.plot(theta_data, label='Тангаж')  # Added label
    plt.ylabel('Тангаж')
    plt.legend()

    plt.subplot(313)
    plt.plot(psi_data, label='Рысканье')  # Added label
    plt.ylabel('Рысканье')
    plt.legend()

    plt.figure()
    plt.subplot(311)
    plt.plot(x_data, label='X')  # Added label
    plt.ylabel('X')
    plt.legend()

    plt.subplot(312)
    plt.plot(y_data, label='Y')  # Added label
    plt.ylabel('Y')
    plt.legend()

    plt.subplot(313)
    plt.plot(z_data, label='Z')  # Added label
    plt.ylabel('Z')
    plt.legend()

    # Plot the additional data
    plt.figure()

    plt.subplot(411)
    plt.plot(distance_data, label='Расстояние до дрона')
    plt.ylabel('Расстояние (м)')
    plt.legend()

    plt.subplot(412)
    plt.plot(time_to_approach_data, label='Время сближения')
    plt.ylabel('Время (с)')
    plt.legend()

    plt.subplot(413)
    plt.plot(speed_data, label='Скорость дрона')
    plt.ylabel('Скорость (м/с)')
    plt.legend()

    plt.subplot(414)
    plt.plot(offset_x_data, label='Смещение X')
    plt.plot(offset_y_data, label='Смещение Y')
    plt.xlabel('Шаг времени')
    plt.ylabel('Смещение (м)')
    plt.legend()

    plt.show()