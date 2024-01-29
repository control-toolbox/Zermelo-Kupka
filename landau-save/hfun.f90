! Landau
subroutine hfun(x, p, c, h)

    double precision, intent(in)  :: x(2), p(2), c(2)
    double precision, intent(out) :: h

    ! local variables
    double precision :: r, th, pr, pth
    double precision :: alpha, delta, beta, nu1, nu2, mu1, mu2, m2

    r   = x(1)
    th  = x(2)

    pr  = p(1)
    pth = p(2)

    alpha = c(1)
    delta = c(2)

    beta = delta*cos(r)
    nu1  = -alpha*sin(r)
    nu2  = 1d0
    mu1  = beta*nu1
    mu2  = beta*nu2

    m2   = sin(r)**2

    h   = mu1*pr + mu2*pth + sqrt(pr**2 + pth**2/m2)

end subroutine hfun