
#ifndef PTR_TYPES_H_INCLUDED
#define PTR_TYPES_H_INCLUDED

#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/make_shared.hpp>

namespace scm {

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

} // namespace scm

#endif // PTR_TYPES_H_INCLUDED
