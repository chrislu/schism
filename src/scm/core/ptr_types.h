
#ifndef PTR_TYPES_H_INCLUDED
#define PTR_TYPES_H_INCLUDED

#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

namespace scm {
//namespace core {

using boost::scoped_ptr;
using boost::scoped_array;
using boost::shared_ptr;
using boost::shared_array;
using boost::weak_ptr;
using boost::intrusive_ptr;

//} // namespace core
} // namespace scm

#endif // PTR_TYPES_H_INCLUDED
