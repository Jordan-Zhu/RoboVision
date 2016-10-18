
# Written 10/1/2016

def relative_pos(line1, line2):
    if line1[10] != 13:
        line_ref = line1
        second = line2
    else:
        line_ref = line2
        second = line1

    m = line_ref[5]
    a = -line_ref[6]
    dir_obj = (line_ref[10] == 8)
    dir_line_ver = abs(m) >= 1

    if a == 90 or a == -90:
        dir_dec = (line_ref[1] - second[1]) > 0
    else:
        hr = line_ref[0] - m * line_ref[1]
        hl = second[0] - m * second[1]
        dir_dec = (hr - hl) > 0
        if m >= 1:
            # And what is this =~ symbol too
            dir_dec = ~dir_dec

    # if the second line was discontinuity too, another condition is needed to
    # check, otherwise if the direction of object and second lines are equal,
    # flag is true.

    if second[10] != 12 and (dir_obj == dir_dec):
        flag_out = second[10] != line_ref[10]
    else:
        # boolean result
        flag_out = (dir_obj == dir_dec)

    return flag_out