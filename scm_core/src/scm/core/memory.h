
#ifndef SCM_CORE_MEMORY_H_INCLUDED
#define SCM_CORE_MEMORY_H_INCLUDED

#if 1

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

#endif // SCM_CORE_MEMORY_H_INCLUDED
