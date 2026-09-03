begin
    y_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val = y_lvl_2.val
    A_lvl = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[1]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    x_lvl = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[2]).tns.bind.lvl
    x_lvl_stop = x_lvl.shape
    x_lvl_2 = x_lvl.lvl
    x_lvl_2_val = x_lvl_2.val
    x_lvl_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl_stop) != $(A_lvl_stop))"))
    Finch.resize_if_smaller!(y_lvl_2_val, A_lvl_2_stop)
    Finch.fill_range!(y_lvl_2_val, 0.0, 1, A_lvl_2_stop)
    phase_stop = min(x_lvl_stop, min(64, x_lvl_stop))
    if phase_stop >= 1
        for j_5 = 1:phase_stop
            x_lvl_q = (1 - 1) * x_lvl_stop + j_5
            x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
            A_lvl_2_origin = 1 + (j_5 - 1) * 5
            phase_start_3 = max(1, 1 + min((1 + (j_5 - 1)) - 1, A_lvl_2_stop))
            phase_stop_3 = min(A_lvl_2_stop, min(((1 + (j_5 - 1)) + 5) - 1, A_lvl_2_stop))
            if phase_stop_3 >= phase_start_3
                for i_6 = phase_start_3:phase_stop_3
                    y_lvl_q = (1 - 1) * A_lvl_2_stop + i_6
                    A_lvl_2_q = A_lvl_2_origin + (i_6 - (1 + (j_5 - 1)))
                    A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                    y_lvl_2_val[y_lvl_q] = x_lvl_2_val_2 * A_lvl_3_val_2 + y_lvl_2_val[y_lvl_q]
                end
            end
        end
    end
    resize!(y_lvl_2_val, A_lvl_2_stop)
    (y = Tensor((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(y_lvl_2_val), A_lvl_2_stop)),)
end