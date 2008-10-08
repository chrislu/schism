
#if 0
#ifndef CVAR_H_INCLUDED
#define CVAR_H_INCLUDED

#include <algorithm>
#include <string>
#include <typeinfo>

#include <scm/core/platform/platform.h>

#include <scm/core/script/script_system.h>


namespace scm {
namespace con {

// global variables
// registered and accessible through the console
// added to the console dictionary

class __scm_export(core) cvar
{
public:
    typedef enum {
        CVAR_T_UNASSIGNED   = 0x00,

        CVAR_T_NUMBER       = 0x01,
        CVAR_T_STRING       = 0x04
    } cvar_type;

    typedef script::script_system_interface::number_t   number_t;

public:
    cvar();
    cvar(const cvar& var);
    cvar& operator=(const cvar& var);
    virtual ~cvar();

    // set
    void                    set_value(number_t v);
    void                    set_value(const std::string& v);

    // get
    number_t                get_number_value() const { return (_number_data); }
    const std::string&      get_string_value() const { return (_string_data); }

    // type
    cvar_type               get_type() const { return (_current_type); }

    // equals
    bool                    equals(const cvar& rhs) const;

protected:
    number_t                _number_data;
    std::string             _string_data;

    cvar_type               _current_type;

    void                    swap(cvar& rhs);

private:
    // friend operators
    friend bool             operator==(const cvar&, const cvar&);

    friend bool             operator==(const cvar&, number_t);
    friend bool             operator==(const cvar&, const std::string&);

    friend bool             operator==(number_t, const cvar&);
    friend bool             operator==(const std::string&, const cvar&);

    friend __scm_export(core) std::ostream&    operator<<(std::ostream&, const cvar&);

    friend void             std::swap<cvar>(cvar& lhs, cvar& rhs);
}; // class cvar

} // namespace con
} // namespace scm

#include "cvar.inl"

#endif // CVAR_H_INCLUDED

#endif
