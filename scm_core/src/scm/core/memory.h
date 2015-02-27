
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_MEMORY_H_INCLUDED
#define SCM_CORE_MEMORY_H_INCLUDED

#include <scm/core/numeric_types.h>

#if 1

#include <memory>

#include <boost/array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace scm {

using boost::array;

using boost::scoped_ptr;
using boost::scoped_array;
using boost::shared_ptr;
using boost::shared_array;
using boost::weak_ptr;
using boost::intrusive_ptr;

using boost::make_shared;
using boost::allocate_shared;

using boost::static_pointer_cast;
using boost::const_pointer_cast;
using boost::dynamic_pointer_cast;

using boost::enable_shared_from_this;

using std::unique_ptr;

} // namespace scm

#else

#include <array>
#include <memory>

#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>
#include <boost/intrusive_ptr.hpp>

namespace scm {

using boost::scoped_ptr;
using boost::scoped_array;
using boost::shared_array;
using boost::intrusive_ptr;

using std::array;

using std::shared_ptr;
using std::weak_ptr;
using std::unique_ptr;

using std::make_shared;
using std::allocate_shared;

using std::static_pointer_cast;
using std::const_pointer_cast;
using std::dynamic_pointer_cast;

using std::enable_shared_from_this;

} // namespace scm

// fix to get boost::bind with tr1::shared_ptr working
namespace boost {

template<class T> T * get_pointer(std::shared_ptr<T> const & p) {
    return p.get();
}

} // namespace boost

#endif

namespace scm {

inline
uintptr_t
align_address(const void*const p, const uintptr_t a) {
    const uintptr_t p8 = reinterpret_cast<const uintptr_t>(p);
    const uintptr_t r  = p8 % a;
    //return ((p8 + a - 1) / a) * a;
    return p8 + (r == 0 ? 0 : a - r);
}

inline
uintptr_t
align_address(const uintptr_t p, const uintptr_t a) {
    return round_to_multiple(p, a);
}

} // namespace scm

#endif // SCM_CORE_MEMORY_H_INCLUDED
