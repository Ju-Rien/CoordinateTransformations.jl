#############################
### 2D Coordinate systems ###
#############################
"""
    Polar{T,A}(r::T, θ::A)

2D polar coordinates
"""
struct Polar{T,A}
    r::T
    θ::A

    Polar{T, A}(r, θ) where {T, A} = new(r, θ)
end

function Polar(r, θ)
    r2, θ2 = promote(r, θ)

    return Polar{typeof(r2), typeof(θ2)}(r2, θ2)
end

"""
    Polar{T,T}(x::AbstractVector)

2D polar coordinates from an AbstractVector of length 2
"""
function Polar(x::AbstractVector)
    return PolarFromCartesian()(x)
end

Base.show(io::IO, x::Polar) = print(io, "Polar(r=$(x.r), θ=$(x.θ) rad)")
Base.isapprox(p1::Polar, p2::Polar; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...)

"""
    PolarFromCartesian()

Transformation from `AbstractVector` of length 2 to `Polar` type
"""
struct PolarFromCartesian <: Transformation; end
"""
    CartesianFromPolar()

Transformation from `Polar` type to `SVector{2}` type
"""
struct CartesianFromPolar <: Transformation; end

Base.show(io::IO, trans::PolarFromCartesian) = print(io, "PolarFromCartesian()")
Base.show(io::IO, trans::CartesianFromPolar) = print(io, "CartesianFromPolar()")

function (::PolarFromCartesian)(x::AbstractVector)
    length(x) == 2 || error("Polar transform takes a 2D coordinate")

    Polar(hypot(x[1], x[2]), atan(x[2], x[1]))
end

function transform_deriv(::PolarFromCartesian, x::AbstractVector)
    length(x) == 2 || error("Polar transform takes a 2D coordinate")

    r = hypot(x[1], x[2])
    f = x[2] / x[1]
    c = one(eltype(x))/(x[1]*(one(eltype(x)) + f*f))
    @SMatrix [ x[1]/r    x[2]/r ;
              -f*c       c      ]
end
transform_deriv_params(::PolarFromCartesian, x::AbstractVector) = error("PolarFromCartesian has no parameters")

function (::CartesianFromPolar)(x::Polar)
    s,c = sincos(x.θ)
    SVector(x.r * c, x.r * s)
end
function transform_deriv(::CartesianFromPolar, x::Polar)
    sθ, cθ = sincos(x.θ)
    @SMatrix [cθ  -x.r*sθ ;
              sθ   x.r*cθ ]
end
transform_deriv_params(::CartesianFromPolar, x::Polar) = error("CartesianFromPolar has no parameters")

Base.inv(::PolarFromCartesian) = CartesianFromPolar()
Base.inv(::CartesianFromPolar) = PolarFromCartesian()

compose(::PolarFromCartesian, ::CartesianFromPolar) = IdentityTransformation()
compose(::CartesianFromPolar, ::PolarFromCartesian) = IdentityTransformation()

# For convenience
Base.convert(::Type{Polar}, v::AbstractVector) = PolarFromCartesian()(v)
@inline Base.convert(::Type{V}, p::Polar) where {V <: AbstractVector} = convert(V, CartesianFromPolar()(p))
@inline Base.convert(::Type{V}, p::Polar) where {V <: StaticVector} = convert(V, CartesianFromPolar()(p))


#############################
### 3D Coordinate Systems ###
#############################
"""
    Spherical(r, θ, ϕ)

3D spherical coordinates

There are many Spherical coordinate conventions and this library uses a somewhat exotic one.
Given a vector `v` with Cartesian coordinates `xyz`, let `v_xy = [x,y,0]` be the
orthogonal projection of `v` on the `xy` plane.

* `r` is the radius. It is given by `norm(v, 2)`.
* `θ` is the azimuth. It is the angle from the x-axis to `v_xy`
* `ϕ` is the latitude. It is the angle from `v_xy` to `v`.

```jldoctest
julia> v = randn(3);

julia> sph = SphericalFromCartesian()(v);

julia> r = sph.r; θ = sph.θ; ϕ = sph.ϕ;

julia> v ≈ [r * cos(θ) * cos(ϕ), r * sin(θ) * cos(ϕ), r * sin(ϕ)]
true
```
"""
struct Spherical{T,A}
    r::T
    θ::A
    ϕ::A

    Spherical{T, A}(r, θ, ϕ) where {T, A} = new(r, θ, ϕ)
end

function Spherical(r, θ, ϕ)
    r2, θ2, ϕ2 = promote(r, θ, ϕ)

    return Spherical{typeof(r2), typeof(θ2)}(r2, θ2, ϕ2)
end

function Spherical(x::AbstractVector)
    return SphericalFromCartesian()(x)
end

Base.show(io::IO, x::Spherical) = print(io, "Spherical(r=$(x.r), θ=$(x.θ) rad, ϕ=$(x.ϕ) rad)")
Base.isapprox(p1::Spherical, p2::Spherical; kwargs...) =
    isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.ϕ, p2.ϕ; kwargs...)

"""
    Cylindrical(r, θ, z)

3D cylindrical coordinates
"""
struct Cylindrical{T,A}
    r::T
    θ::A
    z::T

    Cylindrical{T, A}(r, θ, z) where {T, A} = new(r, θ, z)
end

function Cylindrical(r, θ, z)
    r2, θ2, z2 = promote(r, θ, z)

    return Cylindrical{typeof(r2), typeof(θ2)}(r2, θ2, z2)
end

function Cylindrical(x::AbstractVector)
    return CylindricalFromCartesian()(x)
end

Base.show(io::IO, x::Cylindrical) = print(io, "Cylindrical(r=$(x.r), θ=$(x.θ) rad, z=$(x.z))")
Base.isapprox(p1::Cylindrical, p2::Cylindrical; kwargs...) =
    isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.z, p2.z; kwargs...)

"""
    SphericalFromCartesian()

Transformation from 3D point to `Spherical` type
"""
struct SphericalFromCartesian <: Transformation; end
"""
    CartesianFromSpherical()

Transformation from `Spherical` type to `SVector{3}` type
"""
struct CartesianFromSpherical <: Transformation; end
"""
    CylindricalFromCartesian()

Transformation from 3D point to `Cylindrical` type
"""
struct CylindricalFromCartesian <: Transformation; end
"""
    CartesianFromCylindrical()

Transformation from `Cylindrical` type to `SVector{3}` type
"""
struct CartesianFromCylindrical <: Transformation; end
"""
    CylindricalFromSpherical()

Transformation from `Spherical` type to `Cylindrical` type
"""
struct CylindricalFromSpherical <: Transformation; end
"""
    SphericalFromCylindrical()

Transformation from `Cylindrical` type to `Spherical` type
"""
struct SphericalFromCylindrical <: Transformation; end

Base.show(io::IO, trans::SphericalFromCartesian) = print(io, "SphericalFromCartesian()")
Base.show(io::IO, trans::CartesianFromSpherical) = print(io, "CartesianFromSpherical()")
Base.show(io::IO, trans::CylindricalFromCartesian) = print(io, "CylindricalFromCartesian()")
Base.show(io::IO, trans::CartesianFromCylindrical) = print(io, "CartesianFromCylindrical()")
Base.show(io::IO, trans::CylindricalFromSpherical) = print(io, "CylindricalFromSpherical()")
Base.show(io::IO, trans::SphericalFromCylindrical) = print(io, "SphericalFromCylindrical()")

# Cartesian <-> Spherical
function (::SphericalFromCartesian)(x::AbstractVector)
    length(x) == 3 || error("Spherical transform takes a 3D coordinate")

    Spherical(hypot(x[1], x[2], x[3]), atan(x[2], x[1]), atan(x[3], hypot(x[1], x[2])))
end
function transform_deriv(::SphericalFromCartesian, x::AbstractVector)
    length(x) == 3 || error("Spherical transform takes a 3D coordinate")
    T = eltype(x)

    r = hypot(x[1], x[2], x[3])
    rxy = hypot(x[1], x[2])
    fxy = x[2] / x[1]
    cxy = one(T)/(x[1]*(one(T) + fxy*fxy))
    f = -x[3]/(rxy*r*r)

    @SMatrix [ x[1]/r   x[2]/r  x[3]/r;
          -fxy*cxy  cxy     zero(T);
           f*x[1]   f*x[2]  rxy/(r*r) ]
end
transform_deriv_params(::SphericalFromCartesian, x::AbstractVector) =
    error("SphericalFromCartesian has no parameters")

function (::CartesianFromSpherical)(x::Spherical)
    sθ, cθ = sincos(x.θ)
    sϕ, cϕ = sincos(x.ϕ)
    SVector(x.r * cθ * cϕ, x.r * sθ * cϕ, x.r * sϕ)
end
function transform_deriv(::CartesianFromSpherical, x::Spherical{T}) where T
    sθ, cθ = sincos(x.θ)
    sϕ, cϕ = sincos(x.ϕ)
    @SMatrix [cθ*cϕ -x.r*sθ*cϕ -x.r*cθ*sϕ ;
              sθ*cϕ  x.r*cθ*cϕ -x.r*sθ*sϕ ;
              sϕ     zero(T)    x.r * cϕ  ]
end
transform_deriv_params(::CartesianFromSpherical, x::Spherical) =
    error("CartesianFromSpherical has no parameters")

# Cartesian <-> Cylindrical
function (::CylindricalFromCartesian)(x::AbstractVector)
    length(x) == 3 || error("Cylindrical transform takes a 3D coordinate")

    Cylindrical(hypot(x[1], x[2]), atan(x[2], x[1]), x[3])
end

function transform_deriv(::CylindricalFromCartesian, x::AbstractVector)
    length(x) == 3 || error("Cylindrical transform takes a 3D coordinate")
    T = eltype(x)

    r = hypot(x[1], x[2])
    f = x[2] / x[1]
    c = one(T)/(x[1]*(one(T) + f*f))
    @SMatrix [ x[1]/r   x[2]/r   zero(T) ;
              -f*c      c        zero(T) ;
               zero(T)  zero(T)  one(T)  ]
end
transform_deriv_params(::CylindricalFromCartesian, x::AbstractVector) =
    error("CylindricalFromCartesian has no parameters")

function (::CartesianFromCylindrical)(x::Cylindrical)
    sθ, cθ = sincos(x.θ)
    SVector(x.r * cθ, x.r * sθ, x.z)
end
function transform_deriv(::CartesianFromCylindrical, x::Cylindrical{T}) where {T}
    sθ, cθ = sincos(x.θ)
    @SMatrix [cθ      -x.r*sθ  zero(T) ;
              sθ       x.r*cθ  zero(T) ;
              zero(T)  zero(T) one(T)  ]
end
transform_deriv_params(::CartesianFromPolar, x::Cylindrical) =
    error("CartesianFromCylindrical has no parameters")

function (::CylindricalFromSpherical)(x::Spherical)
    sϕ, cϕ = sincos(x.ϕ)
    Cylindrical(x.r*cϕ,x.θ,x.r*sϕ)
end
function transform_deriv(::CylindricalFromSpherical, x::Spherical)
    M1 = transform_deriv(CylindricalFromCartesian(), CartesianFromSpherical()(x))
    M2 = transform_deriv(CartesianFromSpherical(), x)
    return M1*M2
end
transform_deriv_params(::CylindricalFromSpherical, x::Spherical) =
    error("CylindricalFromSpherical has no parameters")

function (::SphericalFromCylindrical)(x::Cylindrical)
    Spherical(hypot(x.r,x.z),x.θ,atan(x.z,x.r))
end
function transform_deriv(::SphericalFromCylindrical, x::Cylindrical)
    M1 = transform_deriv(SphericalFromCartesian(), CartesianFromCylindrical()(x))
    M2 = transform_deriv(CartesianFromCylindrical(), x)
    return M1*M2
end
transform_deriv_params(::SphericalFromCylindrical, x::Cylindrical) =
    error("SphericalFromCylindrical has no parameters")

Base.inv(::SphericalFromCartesian)   = CartesianFromSpherical()
Base.inv(::CartesianFromSpherical)   = SphericalFromCartesian()
Base.inv(::CylindricalFromCartesian) = CartesianFromCylindrical()
Base.inv(::CartesianFromCylindrical) = CylindricalFromCartesian()
Base.inv(::CylindricalFromSpherical) = SphericalFromCylindrical()
Base.inv(::SphericalFromCylindrical) = CylindricalFromSpherical()

# Inverse composition
compose(::SphericalFromCartesian,   ::CartesianFromSpherical)   = IdentityTransformation()
compose(::CartesianFromSpherical,   ::SphericalFromCartesian)   = IdentityTransformation()
compose(::CylindricalFromCartesian, ::CartesianFromCylindrical) = IdentityTransformation()
compose(::CartesianFromCylindrical, ::CylindricalFromCartesian) = IdentityTransformation()
compose(::CylindricalFromSpherical, ::SphericalFromCylindrical) = IdentityTransformation()
compose(::SphericalFromCylindrical, ::CylindricalFromSpherical) = IdentityTransformation()

# Cyclic compositions
compose(::SphericalFromCartesian,   ::CartesianFromCylindrical) = SphericalFromCylindrical()
compose(::CartesianFromSpherical,   ::SphericalFromCylindrical) = CartesianFromCylindrical()
compose(::CylindricalFromCartesian, ::CartesianFromSpherical)   = CylindricalFromSpherical()
compose(::CartesianFromCylindrical, ::CylindricalFromSpherical) = CartesianFromSpherical()
compose(::CylindricalFromSpherical, ::SphericalFromCartesian)   = CylindricalFromCartesian()
compose(::SphericalFromCylindrical, ::CylindricalFromCartesian) = SphericalFromCartesian()

# For convenience
Base.convert(::Type{Spherical}, v::AbstractVector) = SphericalFromCartesian()(v)
Base.convert(::Type{Cylindrical}, v::AbstractVector) = CylindricalFromCartesian()(v)

Base.convert(::Type{V}, s::Spherical) where {V <: AbstractVector} = convert(V, CartesianFromSpherical()(s))
Base.convert(::Type{V}, c::Cylindrical) where {V <: AbstractVector} = convert(V, CartesianFromCylindrical()(c))
Base.convert(::Type{V}, s::Spherical) where {V <: StaticVector} = convert(V, CartesianFromSpherical()(s))
Base.convert(::Type{V}, c::Cylindrical) where {V <: StaticVector} = convert(V, CartesianFromCylindrical()(c))

Base.convert(::Type{Spherical}, c::Cylindrical) = SphericalFromCylindrical()(c)
Base.convert(::Type{Cylindrical}, s::Spherical) = CylindricalFromSpherical()(s)











##############################
### N-D Coordinate Systems ###
##############################
"""
    Hyperspherical(r, φ1, φ2, ..., φₙ₋₁)
    Hyperspherical(r, Φ::SVector{N-1, φT})
N-D hyperspherical coordinates

This struct represents a point in N-dimensional Euclidean space using hyperspherical coordinates. 
The coordinates consist of a radial coordinate `r` and `N-1` angular coordinates `φ1, φ2, ..., φₙ₋₁`.

Given a vector `v` with Cartesian coordinates `x₁, x₂, ..., xₙ`, the hyperspherical coordinates are defined as follows:

* `r` is the radius, given by `norm(v, 2)`, representing the distance from the origin to the point.
* `φ₁` is the first angular coordinate, representing the angle between the vector and the positive `x₁` axis (`φ₁ = atan(√(x₂² + x₃² + … + xₙ²), x₁)`).
* `φ₂, φ₃, ..., φₙ₋₂` are the intermediate angular coordinates, each defined recursively as the angle between the projection of the vector onto the subspace orthogonal to the previous axes and the next coordinate axis (`φₖ = atan(√(xₖ₊₁² + x₃² + … + xₙ²), xₖ)`).
* `φₙ₋₁` is the final angular coordinate, analogous to the azimuthal angle in 3D spherical coordinates, representing the angle between the vector projection on the last two axes (`φ₁ = atan(xₙ, x₁)`).

There are some special cases where the Cartesian to Hyperspherical transform is not unique; φₖ for any k will be ambiguous whenever all of xₖ , xₖ₊₁, …, xₙ are zero; in this case φₖ is to be zero.

The forward and inverse transformations between Cartesian and hyperspherical coordinates are implemented.

```jldoctest
julia> v = randn(SVector{4, Float64});  # A 4D vector

julia> hsph = Hyperspherical(v);  # Convert to 4D hyperspherical coordinates

julia> r = hsph.r; φ₁, φ₂, φ₃ = hsph.Φ...;

julia> v_reconstructed = CartesianFromHyperspherical()(hsph);

julia> v ≈ v_reconstructed  # Verify the inverse transformation
true
```
"""
struct Hyperspherical{T, A, N}
    r::T
    Φ::SVector{N_minus_1, A} where {N_minus_1}

    function Hyperspherical(r::T, Φ::SVector{N_minus_1, A}) where {T, A, N_minus_1}
        if N_minus_1 <= 1
            throw(ArgumentError("N must be greater than 1"))
        end
        
        new{T, A, N_minus_1+1}(r, Φ)
    end
end

function Hyperspherical(args...)
    @assert length(args)>0

    # Dispatch Hyperspherical(x::AbstractVector) should capture the case where args has length > 1
    #   but is not a Tuple.
    if length(args)==1
        return args[1] # Trivial case, no need for Hyperspherical
    elseif length(args)==2
        return Polar(args...)
    elseif length(args)==3
        return Spherical(args...)
    end

    args = promote(args...)
    println(typeof(SVector(args[2:end]))) # TODO: Remove
    return Hyperspherical(args[1], SVector(args[2:end]))
end

function Hyperspherical(x::AbstractVector)
    return HypersphericalFromCartesian()(x)
end

Base.show(io::IO, x::Hyperspherical) = print(io, "Hyperspherical(r=$(x.r), Φ=[ $(reduce((str1,str2)->"$(str1), $(str2)", string.(x.Φ))) ] rad)")
Base.isapprox(p1::Hyperspherical, p2::Hyperspherical; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.Φ, p2.Φ; kwargs...)

"""
    HypersphericalFromCartesian()

Transformation from ND point to `Hyperspherical` type
"""
struct HypersphericalFromCartesian <: Transformation; end
"""
    CartesianFromHyperspherical()

Transformation from `Hyperspherical` type to `SVector{N}` type
"""
struct CartesianFromHyperspherical <: Transformation; end


####   //\   ####   //\   ####   //\   ###############################   //\   ####   //\   ####   //\   #####
####  // \\  ####  // \\  ####  // \\  ####                       ####  // \\  ####  // \\  ####  // \\  #####
#### //   \\ #### //   \\ #### //   \\ ####                       #### //   \\ #### //   \\ #### //   \\ #####
####//     \\####//     \\####//     \\###############################//     \\####//     \\####//     \\#####
##############################################################################################################
##############################################################################################################
## ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ###
######    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####
#####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    #####
##############################################################################################################
#
#     WW      WW      OOOO      RRRR       KKK KKK      SSSS      PPPPP        AAAA       CCCC     EEEEE     #
#     WW      WW     OO  OO     RR  RR     KK KK       S          PP   PP     AA  AA     CC        EE        #
#     WW  WW  WW     OO  OO     RRRR       KKKK         SSSS      PPPPP       AAAAAA     CC        EEEEE     #
#     WW  WW  WW     OO  OO     RR RR      KK KK            S     PP          AA  AA     CC        EE        #
#      WW    WW       OOOO      RR  RR     KK  KK       SSSS      PP          AA  AA      CCCC     EEEEE     #
#
##############################################################################################################
####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ######
###    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    #### ##
##    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####  ##
##############################################################################################################
# v workspace v

# function (::HypersphericalFromCartesian)(x::AbstractVector)
    
#     N = length(x)
#     r = norm(x)
#     φ = NTuple{N-1, eltype(x)}(undef)

#     φ[1] = atan2(norm(x[2:end]), x[1])
#     for i in 2:(N-2)
#         if norm(x[i:end]) == 0
#             φ[i] = zero(eltype(x))
#         else
#             φ[i] = atan2(norm(x[i+1:end]), x[i])
#         end
#     end
#     φ[N-1] = atan2(x[end], x[end-1])

#     return Hyperspherical(r, φ)
# end

function (::HypersphericalFromCartesian)(x::AbstractVector)
    HypersphericalFromCartesian()(SVector{length(x),eltype(x)}(x))
end

function (::HypersphericalFromCartesian)(x::SVector{n, T}) where {n, T}

    if length(x)==1
        return x
    elseif length(x)==2
        return Polar(x)
    elseif length(x)==3
        return Spherical(x)
    end

    radius = norm(x)
    floatType = eltype(radius)
    angles = SizedArray(zeros(SVector{n-1, floatType}))

    only_nulls = false

    for i in n:-1:1
        if i == 1
            continue
        elseif i == n
            if (x[n] ≠ 0 || x[n-1] ≠ 0)
                angles[n-1] = 2*atan(x[n]/( x[n-1] + norm([ x[n-1], x[n] ]) ))
                only_nulls = false
            else
                angles[n-1] = floatType(0)
                only_nulls = true
            end
        else
            if only_nulls == true && x[i] == 0 && x[i-1] ≠ 0
                if x[i-1] < 0
                    angles[i-1] = floatType(pi)
                elseif x[i-1] > 0
                    angles[i-1] = floatType(0)
                else
                    throw("UNREACHABLE")
                end;
                only_nulls = false
            elseif only_nulls == true && x[i] == 0 && x[i-1] == 0
                angles[i-1] = floatType(0)
            else
                angles[i-1] = atan(norm(x[i:n])/x[i-1])
            end
        end
    end

    Hyperspherical(radius, SVector(angles))
end

# ############# ?
# function transform_deriv(::HypersphericalFromCartesian, x::AbstractVector)
#     # TODO
#     length(x) == 3 || error("Spherical transform takes a 3D coordinate")
#     T = eltype(x)

#     r = hypot(x[1], x[2], x[3])
#     rxy = hypot(x[1], x[2])
#     fxy = x[2] / x[1]
#     cxy = one(T)/(x[1]*(one(T) + fxy*fxy))
#     f = -x[3]/(rxy*r*r)

#     @SMatrix [ x[1]/r   x[2]/r  x[3]/r;
#           -fxy*cxy  cxy     zero(T);
#            f*x[1]   f*x[2]  rxy/(r*r) ]
# end


# ############# ?
# # TODO
# transform_deriv_params(::HypersphericalFromCartesian, x::AbstractVector) =
#     error("SphericalFromCartesian has no parameters")


# function (::CartesianFromHyperspherical)(h::Hyperspherical)
#     r, φ = h.r, h.φ
#     N = length(φ) + 1
#     x = MVector{N, eltype(r)}(undef)

#     sin_prod = r
#     for i in 1:(N-1)
#         if i == N-1
#             x[i] = sin_prod * cos(φ[i])
#             x[i+1] = sin_prod * sin(φ[i])
#         else
#             x[i] = sin_prod * cos(φ[i])
#             sin_prod *= sin(φ[i])
#         end
#     end

#     return SVector{N}(x)
# end
    

# ############# ?
# function transform_deriv(::CartesianFromHyperspherical, x::Hyperspherical{T}) where T
#     sθ, cθ = sincos(x.θ)
#     sϕ, cϕ = sincos(x.ϕ)
#     @SMatrix [cθ*cϕ -x.r*sθ*cϕ -x.r*cθ*sϕ ;
#               sθ*cϕ  x.r*cθ*cϕ -x.r*sθ*sϕ ;
#               sϕ     zero(T)    x.r * cϕ  ]
# end

# ############# ?
# transform_deriv_params(::CartesianFromHyperspherical, x::Hyperspherical) =
#     error("CartesianFromHyperspherical has no parameters")

# # Interface
# length(x::Hyperspherical) = length(φ) + 1

# # Inverse composition
# compose(::HypersphericalFromCartesian, ::CartesianFromHyperspherical) = IdentityTransformation()
# compose(::CartesianFromHyperspherical, ::HypersphericalFromCartesian) = IdentityTransformation()


# ^ workspace ^
##############################################################################################################
## ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ###
######    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####
#####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    #####
##############################################################################################################
##############################################################################################################
####\\     //####\\     //####\\     //################################\\     //####\\     //####\\     //####
#### \\   // #### \\   // #### \\   // ####                        #### \\   // #### \\   // #### \\   // ####
####  \\ //  ####  \\ //  ####  \\ //  ####                        ####  \\ //  ####  \\ //  ####  \\ //  ####
####   \\/   ####   \\/   ####   \\/   ################################   \\/   ####   \\/   ####   \\/   ####

