
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <boost/functional/hash.hpp>

namespace scm {
namespace res {

inline std::size_t hash_value(const resource_base& ref)
{
    return (ref.hash_value());
}

template<class res_desc>
inline std::size_t hash_value(const resource<res_desc>& ref)
{
    return (ref.hash_value());
}

template <class res_desc>
resource<res_desc>::resource(const res_desc& desc)
: _descriptor(desc)
{
}

template <class res_desc>
resource<res_desc>::~resource()
{
}

template <class res_desc>
const res_desc&
resource<res_desc>::get_descriptor() const
{
    return (_descriptor);
}

template <class res_desc>
resource_base::hash_type
resource<res_desc>::hash_value() const
{
    return (_descriptor.hash_value());
}

} // namespace res
} // namespace scm
