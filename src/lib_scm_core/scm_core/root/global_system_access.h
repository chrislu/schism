
#ifndef GLOBAL_SYSTEM_ACCESS_H_INCLUDED
#define GLOBAL_SYSTEM_ACCESS_H_INCLUDED

#include <boost/noncopyable.hpp>

namespace scm
{
    namespace core
    {
        class system_root_interface;

        template <class instance_t>
        struct global_system_access
        {
        private:
            // nested class definition
            template <class instance_t_>
            class global_system_access_ : boost::noncopyable
            {
            public:
                explicit global_system_access_();
                virtual ~global_system_access_();

                instance_t_&                 get() const;
                instance_t_&                 operator*() const;
                instance_t_*const            get_ptr() const;
                instance_t_*const            operator->() const;

            private:
                void                        set_instance(instance_t_*const) const;
                mutable instance_t_*         _instance;

                // only system root has access to the write functions
                friend class scm::core::system_root_interface;
            }; // class global_system_access_
        public:
            typedef const global_system_access_<instance_t> type;
        }; // struct global_system_access

    } // namespace core
} // namespace scm

#include "global_system_access.inl"

#endif // GLOBAL_SYSTEM_ACCESS_H_INCLUDED

