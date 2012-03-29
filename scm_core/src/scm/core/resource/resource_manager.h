
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef RESOURCE_MANAGER_H_INCLUDED
#define RESOURCE_MANAGER_H_INCLUDED

#if 0

#include <cstddef>
#include <map>

#include <boost/shared_ptr.hpp>

#include <scm/core/resource/resource.h>
#include <scm/core/resource/resource_pointer.h>

#include <scm/core/sys_interfaces.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace res {

class resource_pointer_base;

class __scm_export(core) resource_manager_base : public system,
                                                 public scm::enable_shared_from_this<resource_manager_base>
{
protected:
    typedef boost::shared_ptr<resource_base>    res_ptr_type;
    typedef std::map<resource_base::hash_type,
                     std::pair<res_ptr_type,
                               std::size_t> >   resource_container;

public:
    resource_manager_base();
    virtual ~resource_manager_base();

    using system::initialize;
    virtual bool                        shutdown();

    virtual bool                        is_loaded(const resource_pointer_base& /*inst*/)   const;
    virtual bool                        is_loaded(const resource_base::hash_type /*hash*/) const;

    void                                register_instance(const resource_pointer_base& /*inst*/);
    void                                release_instance(const resource_pointer_base&  /*inst*/);

protected:
    void                                insert_instance_values(const resource_base::hash_type /*hash*/,
                                                               const res_ptr_type&            /*ptr*/);

    void                                clear_resources();

    resource_container                  _resources;

private:

}; // class resource_manager_base


template<class res_type>
class resource_manager : public resource_manager_base
{
public:
    typedef res_type                            resource_type;
    typedef typename res_type::descriptor_type  resource_descriptor_type;

public:
    resource_manager();
    virtual ~resource_manager();

    virtual bool                        is_loaded(const resource_descriptor_type& /*desc*/) const;

    virtual resource_pointer<res_type>  retrieve_instance(const resource_base::hash_type /*hash*/);
    virtual resource_pointer<res_type>  retrieve_instance(const resource_descriptor_type& /*desc*/);
    virtual resource_pointer<res_type>  create_instance(const resource_descriptor_type&   /*desc*/);

    res_type&                           to_resource(resource_pointer_base&           /*ref*/) const;
    const res_type&                     to_resource(const resource_pointer_base&     /*ref*/) const;
    resource_pointer<res_type>&         to_resource_ptr(resource_pointer_base&       /*ref*/) const;
    const resource_pointer<res_type>&   to_resource_ptr(const resource_pointer_base& /*ref*/) const;

protected:

private:

}; // class resource_manager

} // namespace res
} // namespace scm

#include "resource_manager.inl"

#include <scm/core/utilities/platform_warning_enable.h>


#endif // 0

#endif // RESOURCE_MANAGER_H_INCLUDED
