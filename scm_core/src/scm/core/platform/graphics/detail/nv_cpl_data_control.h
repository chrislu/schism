
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef NV_CPL_DATA_CONTROL_H_INCLUDED
#define NV_CPL_DATA_CONTROL_H_INCLUDED


namespace scm {
namespace platform {
namespace detail {

struct nv_cpl_data_control
{
private:
    typedef bool (*nv_cpl_get_data_int_ptr)(long, long*);
    typedef bool (*nv_cpl_set_data_int_ptr)(long, long);

public:
    nv_cpl_data_control() : _get_data_int(0), _set_data_int(0) {}

    bool    initialize_cpl_control();
    void    close_cpl_control();

    nv_cpl_get_data_int_ptr _get_data_int;
    nv_cpl_set_data_int_ptr _set_data_int;


}; // class nv_cpl_data_control

} // namespace detail
} // namespace platform
} // namespace scm

#endif // NV_CPL_DATA_CONTROL_H_INCLUDED
