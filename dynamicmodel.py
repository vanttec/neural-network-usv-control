import numpy as np

X_u_dot = -2.25
Y_v_dot = -23.13
Y_r_dot = -1.31
N_v_dot = -16.41
N_r_dot = -2.79
Xuu = 0
Yvv = -99.99
Yvr = -5.49
Yrv = -5.49
Yrr = -8.8
Nvv = -5.49
Nvr = -8.8
Nrv = -8.8
Nrr = -3.49
m = 30
Iz = 4.1
B = 0.41
c = 0.78

rate = 100
integral_step = 0.01


def model(Tport, Tstbd, upsilon=np.zeros(3, dtype=np.float),
          eta=np.zeros(3, dtype=np.float)):
    upsilon_dot_last = np.array([0.0, 0.0, 0.0])
    eta_dot_last = np.array([0.0, 0.0, 0.0])

    Xu = -0.25
    Xuu = 0
    if(abs(upsilon[0] > 1.2)):
        Xu = 64.55
        Xuu = -70.92

    Yv = 0.5*(-40*1000*abs(upsilon[1])) * \
        (1.1+0.0045*(1.01/0.09) - 0.1*(0.27/0.09)+0.016*(np.power((0.27/0.09), 2)))
    Yr = 6*(-3.141592*1000)*np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
    Nv = 0.06*(-3.141592*1000) * \
        np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01
    Nr = 0.02*(-3.141592*1000) * \
        np.sqrt(np.power(upsilon[0], 2)+np.power(upsilon[1], 2))*0.09*0.09*1.01*1.01

    M = np.array([[m - X_u_dot, 0, 0],
                  [0, m - Y_v_dot, 0 - Y_r_dot],
                  [0, 0 - N_v_dot, Iz - N_r_dot]])

    T = np.array([Tport + c*Tstbd, 0, 0.5*B*(Tport - c*Tstbd)])

    CRB = np.array([[0, 0, 0 - m * upsilon[1]],
                    [0, 0, m * upsilon[0]],
                    [m * upsilon[1], 0 - m * upsilon[0], 0]])

    CA = np.array([[0, 0, 2 * ((Y_v_dot*upsilon[1]) + ((Y_r_dot + N_v_dot)/2) * upsilon[2])],
                   [0, 0, 0 - X_u_dot * m * upsilon[0]],
                   [2*(((0 - Y_v_dot) * upsilon[1]) - ((Y_r_dot+N_v_dot)/2) * upsilon[2]), X_u_dot * m * upsilon[0], 0]])

    C = CRB + CA

    Dl = np.array([[0 - Xu, 0, 0],
                   [0, 0 - Yv, 0 - Yr],
                   [0, 0 - Nv, 0 - Nr]])

    Dn = np.array([[Xuu * abs(upsilon[0]), 0, 0],
                   [0, Yvv * abs(upsilon[1]) + Yvr * abs(upsilon[2]), Yrv *
                    abs(upsilon[1]) + Yrr * abs(upsilon[2])],
                   [0, Nvv * abs(upsilon[1]) + Nvr * abs(upsilon[2]), Nrv * abs(upsilon[1]) + Nrr * abs(upsilon[2])]])

    D = Dl - Dn

    upsilon_dot = np.matmul(np.linalg.inv(M), (T - np.matmul(C, upsilon) - np.matmul(D, upsilon)))
    upsilon = (integral_step) * (upsilon_dot + upsilon_dot_last)/2 + upsilon  # integral
    upsilon_dot_last = upsilon_dot

    J = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0],
                  [np.sin(eta[2]), np.cos(eta[2]), 0],
                  [0, 0, 1]])

    eta_dot = np.matmul(J, upsilon)  # transformation into local reference frame
    eta = (integral_step)*(eta_dot+eta_dot_last)/2 + eta  # integral
    eta_dot_last = eta_dot

    if (abs(eta[2]) > np.pi):
        eta[2] = (eta[2]/abs(eta[2]))*(abs(eta[2])-2*np.pi)

    return upsilon, eta
