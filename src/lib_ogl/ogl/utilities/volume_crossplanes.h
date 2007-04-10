
#ifndef VOLUME_CROSSPLANES_H_INCLUDED
#define VOLUME_CROSSPLANES_H_INCLUDED

namespace gl
{
    class volume_crossplanes
    {
    public:
        volume_crossplanes();
        virtual ~volume_crossplanes();

        bool                initialize();
        void                render() const;

        void                set_slice_x(float x);
        float               get_slice_x() const { return (_slice_x); }

        void                set_slice_y(float y);
        float               get_slice_y() const { return (_slice_y); }

        void                set_slice_z(float z);
        float               get_slice_z() const { return (_slice_z); }

    protected:
    private:
        void                clean_up();

        unsigned            _planes_dlist;
        mutable bool        _planes_dlist_dirty;

        float               _slice_x;
        bool                _slice_x_enabled;

        float               _slice_y;
        bool                _slice_y_enabled;

        float               _slice_z;
        bool                _slice_z_enabled;
    };

} // namespace gl

#endif // VOLUME_CROSSPLANES_H_INCLUDED



