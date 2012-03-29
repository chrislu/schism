
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_UTIL_SINGLETON_H_INCLUDED
#define SCM_UTIL_SINGLETON_H_INCLUDED

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/utility.hpp>
#include <boost/thread/mutex.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/memory.h>

namespace scm {

/*
    usage:

    single_test.h

    namespace test {
        class  __scm_export(core) single_test
        {
        public:
            single_test() : _test(1) {}
            int _test;
        };

        typedef scm::singleton<single_test> singleton_test;
    } // namespace test

    single_test.cpp

    namespace test {
    }

    SCM_SINGLETON_PLACEMENT(core, test::single_test)
*/

template<class managed_type>
class singleton
{
public:
    typedef managed_type*               instance_ptr_type;
    //typedef const instance_ptr_type     lease_type;

public:
    static managed_type&                get();

protected:

private:
    static void                         create();
    static void                         destroy();

    static instance_ptr_type            _single_instance;
    static boost::mutex                 _lock;
    static bool                         _dead;

    // private default constructor to prevent use as base class
    // declared never defined
    singleton();
    singleton(const singleton&);
    const singleton& operator=(const singleton&);

}; // class singleton

} // namespace scm

#include "singleton.inl"

#endif // SCM_UTIL_SINGLETON_H_INCLUDED
