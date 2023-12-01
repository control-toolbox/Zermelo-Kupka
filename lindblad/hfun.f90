! Lindblad
subroutine hfun(x, p, d, h)

    double precision, intent(in)  :: x(2), p(2), d
    double precision, intent(out) :: h

    ! local variables
    double precision :: r, th, pr, pth
    double precision :: l, m2, mu

    r   = x(1)
    th  = x(2)

    pr  = p(1)
    pth = p(2)

    l   = 4d0/5d0
    m2  = sin(r)**2/(1d0-l*sin(r)**2)
    mu  = d*sin(2*r)

    h   = mu*pr + sqrt(pr**2 + pth**2/m2)

end subroutine hfun