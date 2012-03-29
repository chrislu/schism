
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef VERSION_H_INCLUDED
#define VERSION_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>

namespace scm {

static const std::string    SCHISM_NAME         = std::string("schism");

static const unsigned       VERSION_MAJOR       = 0;
static const unsigned       VERSION_MINOR       = 7;
static const unsigned       VERSION_REVISION    = 9;

static const std::string    VERSION_TAG         = std::string("devel");
static const std::string    VERSION_NAME        = std::string("metal, it comes from hell");

#if SCM_DEBUG
static const std::string    VERSION_BUILD_TAG   = std::string("debug");
#else
static const std::string    VERSION_BUILD_TAG   = std::string("release");
#endif

#if   SCM_ARCHITECTURE_TYPE == SCM_ARCHITECTURE_32
static const std::string    VERSION_ARCH_TAG   = std::string("x86");
#elif SCM_ARCHITECTURE_TYPE == SCM_ARCHITECTURE_64
static const std::string    VERSION_ARCH_TAG   = std::string("x64");
#endif

// gone names
//   - friend of the night
//   - on the side of the demons
//   - to live is to die

// new name candidates:
//   - flipper died a natural death
//   - they were all in love with dying
//   - lux aeterna
//   - metal, it comes from hell
//   - evil men in the gardens of paradise
//   - on the side of the demons
//   - safe humaity from damnation by bringing the love of god
//   - round them up and execute them - publicly
//   - ...and justice for all
//   - to live is to die
//   - delusional machines
//   - what is the universe gonna come up with next
//   - the shape of things to come
//   - we're no here
//   - hope is emo


} // namespace scm

#endif // VERSION_H_INCLUDED
