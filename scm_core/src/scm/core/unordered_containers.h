
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_UNORDERED_CONTAINERS_H_INCLUDED
#define SCM_CORE_UNORDERED_CONTAINERS_H_INCLUDED

#if 1

#define SCM_CORE_STD_UNORDERED_CONTAINERS   0
#define SCM_CORE_BOOST_UNORDERED_CONTAINERS 1

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

namespace scm {

using boost::unordered_map;
using boost::unordered_multimap;
using boost::unordered_set;
using boost::unordered_multiset;

} // namespace scm

#else

#define SCM_CORE_STD_UNORDERED_CONTAINERS   1
#define SCM_CORE_BOOST_UNORDERED_CONTAINERS 0

#include <unordered_map>
#include <unordered_set>

namespace scm {

using std::unordered_map;
using std::unordered_multimap;
using std::unordered_set;
using std::unordered_multiset;

} // namespace scm

#endif

#endif // SCM_CORE_UNORDERED_CONTAINERS_H_INCLUDED
