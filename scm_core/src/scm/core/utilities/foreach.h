
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_FOREACH_H_INCLUDED
#define SCM_FOREACH_H_INCLUDED

#include <boost/foreach.hpp>

#ifndef BOOST_REVERSE_FOREACH
#error "boost version 1.36 or up required for BOOST_REVERSE_FOREACH"
#endif // BOOST_REVERSE_FOREACH

// evil, but what the hell ;) it looks much nicer
#define foreach BOOST_FOREACH
#define foreach_reverse BOOST_REVERSE_FOREACH

#endif // SCM_FOREACH_H_INCLUDED
