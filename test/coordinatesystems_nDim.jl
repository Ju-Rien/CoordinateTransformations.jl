@testset "nD" begin
    hs_from_cart = HypersphericalFromCartesian()
    cart_from_hs = CartesianFromHyperspherical()
    identity_cart = IdentityTransformation()
    identity_hs = IdentityTransformation()

    # inverses
    @test inv(hs_from_cart) == cart_from_hs
    @test inv(cart_from_s) == s_from_cart

    # composition of inverses
    @test hs_from_cart ∘ cart_from_hs == identity_hs
    @test cart_from_hs ∘ hs_from_cart == identity_cart

    # Hyperspherical <-> Cartesian
    # test all 8 octants of the sphere (for consistency of branch-cuts)

    # # Octant 1
    # xyz = SVector(1.0, 2.0, 3.0)
    # rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test s_from_cart(collect(xyz)) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 2
    # xyz = SVector(-1.0, 2.0, 3.0)
    # rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, 0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 3
    # xyz = SVector(1.0, -2.0, 3.0)
    # rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, 0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 4
    # xyz = SVector(-1.0, -2.0, 3.0)
    # rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, 0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 5
    # xyz = SVector(1.0, 2.0, -3.0)
    # rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, -0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 6
    # xyz = SVector(-1.0, 2.0, -3.0)
    # rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, -0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 7
    # xyz = SVector(1.0, -2.0, -3.0)
    # rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, -0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    # # Octant 8
    # xyz = SVector(-1.0, -2.0, -3.0)
    # rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, -0.9302740141154721)
    # @test s_from_cart(xyz) ≈ rθϕ
    # @test cart_from_s(rθϕ) ≈ xyz

    # xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cart(xyz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cart, xyz)
    # @test m ≈ m_gn

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
    # xyz_gn = cart_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                 # partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                 # partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
    # m = transform_deriv(cart_from_s, rθϕ)
    # @test m ≈ m_gn

    
    # Spherical <-> Cartesian
    # Just composes at the moment, so a single testcase suffices
    # rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
    # rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
    # @test cyl_from_s(rθϕ) ≈ rθz
    # @test s_from_cyl(rθz) ≈ rθϕ

    # rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
    # rθz_gn = cyl_from_s(rθϕ_gn)
    # m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                 # partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                 # partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
    # m = transform_deriv(cyl_from_s, rθϕ)
    # #@test isapprox(m, m_gn; atol = 1e-12)
    # for (m1,m2) in zip(m,m_gn) # Unfortunately, FixedSizeArrays doesn't pass the keyword arguments to isapprox...
        # @test isapprox(m1, m2; atol=1e-12)
    # end

    # rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
    # rθϕ_gn = s_from_cyl(rθz_gn)
    # m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                 # partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                 # partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
    # m = transform_deriv(s_from_cyl, rθz)
    # #@test isapprox(m, m_gn; atol = 1e-12)
    # @test m ≈ m_gn

    @testset "Common types" begin
        wxyz = SVector(1.0, 2.0, 3.0, 4.0)
        wxyz_i = SVector(1, 2, 3, 4)

        # @testset "Hyperspherical" begin
            # rΦ = Hyperspherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)

            # @test hs_from_cart(wxyz) ≈ rΦ
            # @test hs_from_cart(wxyz_i) ≈ rΦ
            # @test hs_from_cart(collect(wxyz)) ≈ rΦ
            # @test cart_from_hs(rΦ) ≈ wxyz

            # hs1 = Hyperspherical(1, 2.0, 3.0)
            # hs2 = Hyperspherical(1.0, 2, 3)
            # hs3 = Hyperspherical{Int,Int}(1, 2, 3)

            # @test typeof(s1.r) == typeof(s1.θ) == typeof(s1.ϕ) == Float64
            # @test typeof(s2.r) == typeof(s2.θ) == typeof(s2.ϕ) == Float64
            # @test typeof(s3.r) == typeof(s3.θ) == typeof(s3.ϕ) == Int
        # end

    end
end
    