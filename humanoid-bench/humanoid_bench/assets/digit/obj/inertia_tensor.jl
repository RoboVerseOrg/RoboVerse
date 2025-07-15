function cuboid_inertia_tensor(m, w, h, d)
    ixx = (1/12)*m*(h^2+d^2)
    ixy = 0.
    ixz = 0.
    iyy = (1/12)*m*(w^2+h^2)
    iyz = 0.
    izz = 0.
    return ixx, ixy, ixz, iyy, iyz, izz
end

function cylinder_inertia_tensor(m, r, h)
    ixx = (1/12)*m*(3r^2 + h^2)
    ixy = 0.
    ixz = 0.
    iyy = (1/12)*m*(3r^2 + h^2)
    iyz = 0.
    izz = 0.5*m*r^2
    return ixx, ixy, ixz, iyy, iyz, izz
end

# cuboid_inertia_tensor(15.028392, 0.2, 0.9, 0.5)
cylinder_inertia_tensor(0.6, 0.1, 0.3)
