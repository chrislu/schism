
#ifndef PLATFORM_H_INCLUDED
#define PLATFORM_H_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/if.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/logical/not.hpp>
#include <boost/preprocessor/detail/is_nullary.hpp>
#include <boost/preprocessor/detail/is_unary.hpp>
#include <boost/preprocessor/detail/is_binary.hpp>
#include <boost/preprocessor/logical/compl.hpp>

#include <scm/core/platform/config.h>

#define SCM_PLATFORM_WINDOWS        1
#define SCM_PLATFORM_LINUX          2
#define SCM_PLATFORM_APPLE          3

#define SCM_COMPILER_MSVC           1
#define SCM_COMPILER_GNUC           2

#define SCM_ENDIAN_LITTLE           1
#define SCM_ENDIAN_BIG              2

#define SCM_ARCHITECTURE_32         1
#define SCM_ARCHITECTURE_64         2

// compiler
#if defined(_MSC_VER)
    #define SCM_COMPILER            SCM_COMPILER_MSVC
    #define SCM_COMPILER_VER        _MSC_VER
#elif defined(__GNUC__)
    #define SCM_COMPILER            SCM_COMPILER_GNUC
    #define SCM_COMPILER_VER        (((__GNUC__)*100) + \
                                    (__GNUC_MINOR__*10) + \
                                    __GNUC_PATCHLEVEL__)
#else
    #pragma error "unknown compiler"
#endif

// platform
#if defined(__WIN32__) || defined(_WIN32)
    #define SCM_PLATFORM            SCM_PLATFORM_WINDOWS
#elif defined(__APPLE_CC__)
    #define SCM_PLATFORM            SCM_PLATFORM_APPLE
#else
    #define SCM_PLATFORM            SCM_PLATFORM_LINUX
#endif

// architecture
#if defined(__x86_64__) || defined(_M_X64)
    #define SCM_ARCHITECTURE_TYPE   SCM_ARCHITECTURE_64
#else
    #define SCM_ARCHITECTURE_TYPE   SCM_ARCHITECTURE_32
#endif

// compiler messages
#define TO_STR_(x)                  #x
#define TO_STR(x)                   TO_STR_(x)
#define todo(msg)                   message(__FILE__ "(" TO_STR(__LINE__) "): " "todo: " #msg)
#define fix_me(msg)                 message(__FILE__ "(" TO_STR(__LINE__) "): " "fix_me: " #msg)
#define warn(msg)                   message(__FILE__ "(" TO_STR(__LINE__) "): " "warning: " #msg)

// windows related
#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    #if SCM_COMPILER == SCM_COMPILER_MSVC

        #define __scm_export(lib) export_(BOOST_PP_EXPAND(BOOST_PP_CAT(SCM_BUILD_LIBRARY_, lib)))
        #define export_(lib) BOOST_PP_IF(BOOST_PP_IS_NULLARY(lib), __declspec(dllexport), __declspec(dllimport))

        #define __scm_private(lib)
    #else
        #pragma error "unsupported compiler"
    #endif

    #ifdef _DEBUG
        #define SCM_DEBUG   1
    #else
        #define SCM_DEBUG   0
    #endif
#endif

// Linux, Apple
#if    SCM_PLATFORM == SCM_PLATFORM_LINUX \
    || SCM_PLATFORM == SCM_PLATFORM_APPLE
    #if SCM_COMPILER_VER >= 400
        // gcc 4.x attribute visibility
        #define __scm_export(lib)   __attribute__ ((visibility("default")))
        #define __scm_private(lib)  __attribute__ ((visibility("hidden")))
    #else
        #define __scm_export(lib)
        #define __scm_private(lib)
    #endif

    #ifdef DEBUG
        #define SCM_DEBUG   1
    #else
        #define SCM_DEBUG   0
    #endif
#endif

#endif // namespace PLATFORM_H_INCLUDED
