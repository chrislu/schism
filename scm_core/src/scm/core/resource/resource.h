
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef RESOURCE_H_INCLUDED
#define RESOURCE_H_INCLUDED

#include <cstddef>

#include <boost/noncopyable.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace res {

class __scm_export(core) resource_base : public boost::noncopyable
{
public:
    typedef std::size_t     hash_type;

public:
    virtual ~resource_base();

    bool                    operator==(const resource_base& rhs) const;

protected:
    resource_base();

    virtual hash_type       hash_value() const = 0;

private:
    friend class resource_manager_base;
    friend std::size_t      hash_value(const resource_base& /*ref*/);

}; // class resource_base

template <class res_desc>
class resource : public resource_base
{
public:
    typedef res_desc        descriptor_type;

public:
    virtual ~resource();

    const res_desc&         get_descriptor() const;

protected:
    resource(const res_desc& /*desc*/);

    hash_type               hash_value() const;

    res_desc                _descriptor;

    template <class des_t>
    friend std::size_t hash_value(const resource<des_t>&);
}; // class resource


} // namespace res
} // namespace scm

#include "resource.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // RESOURCE_H_INCLUDED
