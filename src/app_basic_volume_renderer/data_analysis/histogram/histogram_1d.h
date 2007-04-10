
#ifndef HISTOGRAM_1D_H_INCLUDED
#define HISTOGRAM_1D_H_INCLUDED

#pragma managed(push, off)


#include <cassert>
#include <vector>

#include <data_analysis/value_range.h>

namespace scm
{
    template<typename> class histogram_1d_calculator;


    template<typename val_type>
    class histogram_1d
    {
    public:
        template<typename>
        struct bin {
            bin() : _absolute_amount(0),
                    _relative_amount(.0f) {}

            scm::value_range<val_type>              _value_range;
            unsigned                                _absolute_amount;
            float                                   _relative_amount;

        };

        typedef std::vector<bin<val_type> >         bin_container_t;

    public:
        histogram_1d() : _max_absolute_amount(0),
                        _max_relative_amount(0.0f) {}
        virtual ~histogram_1d() {}

        //const scm::value_range<val_type>&           get_value_range() const     { return (_value_range); }
        typename bin_container_t::size_type         get_num_bins() const        { return (_bins.size()); }
        unsigned                                    get_max_absolute_amount() const { return (_max_absolute_amount); }
        float                                       get_max_relative_amount() const { return (_max_relative_amount); }

        const bin<val_type>&                        operator[](unsigned i) const{ assert(i < _bins.size()); return (_bins[i]); }

    protected:
        bin_container_t                             _bins;

        unsigned                                    _max_absolute_amount;
        float                                       _max_relative_amount;

    private:
        friend class histogram_1d_calculator<val_type>;
    }; // class histogram_1d

} // namespace scm

#pragma managed(pop)

#endif // HISTOGRAM_1D_H_INCLUDED


