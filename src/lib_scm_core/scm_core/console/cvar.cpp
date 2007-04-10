
#include "cvar.h"

#include <limits>
#include <boost/lexical_cast.hpp>

using namespace scm::core;

cvar::cvar()
    : _current_type(CVAR_T_UNASSIGNED),
      _number_data(number_t())
{
}

cvar::cvar(const cvar& var)
    : _current_type(var._current_type),
      _number_data(var._number_data),
      _string_data(var._string_data)
{
}

cvar::~cvar()
{
}

cvar& cvar::operator=(const cvar& rhs)
{
    _current_type = rhs._current_type;
    _number_data  = rhs._number_data;
    _string_data  = rhs._string_data;
    return (*this);
}


void cvar::swap(cvar& rhs)
{
    cvar tmp(*this);
    *this = rhs;
    rhs   = tmp;
}

// set
void cvar::set_value(number_t v)
{
    _number_data    = v;
    _string_data    = boost::lexical_cast<std::string>(_number_data);
    _current_type   = CVAR_T_NUMBER;
}

void cvar::set_value(const std::string& v)
{
    _string_data    = v;
    try {
        _number_data    = boost::lexical_cast<number_t>(_string_data);
    }
    catch (boost::bad_lexical_cast&) {
        _number_data    = std::numeric_limits<number_t>::infinity();
    }
    _current_type   = CVAR_T_STRING;
}

// equals
bool cvar::equals(const cvar& rhs) const
{
    return (   _current_type == rhs._current_type
            && _number_data  == rhs._number_data
            && _string_data  == rhs._string_data);
}

std::ostream& scm::core::operator << (std::ostream& os, const cvar& var) {
    
    switch (var._current_type) {
        case cvar::CVAR_T_NUMBER:       os << var._number_data << " (cvar_type: number)"; break;
        case cvar::CVAR_T_STRING:       os << var._string_data << " (cvar_type: string)"; break;
        case cvar::CVAR_T_UNASSIGNED:   os << "(cvar_type: unassigned)"; break;
    }
    return (os);
}
