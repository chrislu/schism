
#ifndef SYS_INTERFACES_H_INCLUDED
#define SYS_INTERFACES_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm_core/platform/platform.h>

namespace scm
{
    namespace core
    {
        class __scm_export system : public boost::noncopyable
        {
        public:
            system() : _initialized(false) {}
            virtual ~system() {}

            virtual bool            initialize()    = 0;
            virtual bool            shutdown()      = 0;

        protected:
            bool                    _initialized;

        }; // class system

        class __scm_export system_refreshable : public scm::core::system
        {
        public:
            system_refreshable() : system() {}
            virtual ~system_refreshable() {}

            using scm::core::system::initialize;
            using scm::core::system::shutdown;

            virtual bool            frame()         = 0;

        }; // class system_refreshable

    } // namespace system

} // namespace scm

#endif // SYS_INTERFACES_H_INCLUDED

