
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "target.h"

#include <algorithm>

namespace scm {
namespace inp {

target::target(std::size_t id)
  : _id(id),
    _transform(scm::math::mat4f::identity())
{
}

target::~target()
{
}

target::target(const target& ref)
  : _id(ref._id),
    _transform(ref._transform)
{
}

const target& target::operator=(const target& rhs)
{
    _id         = rhs._id;
    _transform  = rhs._transform;

    return (*this);
}

void target::swap(target& ref)
{
    std::swap(_id, ref._id);
    std::swap(_transform, ref._transform);
}

std::size_t target::id() const
{
    return (_id);
}

const scm::math::mat4f& target::transform() const
{
    return (_transform);
}

void target::transform(const scm::math::mat4f& trans)
{
    _transform  = trans;
}

} // namespace inp
} // namespace scm
