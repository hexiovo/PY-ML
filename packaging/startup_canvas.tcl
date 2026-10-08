# The executable manifest enables DPI awareness before Tk creates its window.
# Use real text and canvas shapes; the bundled legacy PNG is never displayed.
set pyml_scale [expr {[winfo fpixels . 1i] / 96.0}]
set pyml_width [expr {int(560 * $pyml_scale)}]
set pyml_height [expr {int(250 * $pyml_scale)}]
.root.canvas delete all
.root.canvas configure -width $pyml_width -height $pyml_height -background #F8FAFF
font create pymlTitle -family {Microsoft YaHei UI} -size [expr {-int(27 * $pyml_scale)}]
font create pymlBody -family {Microsoft YaHei UI} -size [expr {-int(15 * $pyml_scale)}]
.root.canvas create rectangle 1 1 559 249 -fill #F8FAFF -outline #BFDBFE -width 2
.root.canvas create rectangle 32 36 96 100 -fill #16365C -outline #16365C
.root.canvas create line 44 83 44 55 64 74 84 55 84 83 -fill #5EEAD4 -width 5
.root.canvas create text 118 58 -text {正在开启程序中…} -font pymlTitle -fill #16365C -anchor w
.root.canvas create text 118 92 -text {初始化完成后将自动进入工作台} -font pymlBody -fill #475569 -anchor w
.root.canvas create line 32 144 528 144 -fill #DBEAFE
.root.canvas create rectangle 32 173 528 182 -fill #E2E8F0 -outline #E2E8F0
.root.canvas create rectangle 32 173 152 182 -fill #2563EB -outline #2563EB -tag loading
.root.canvas create text 32 218 -text $status_text -font pymlBody -fill #475569 -anchor sw -tag vartext
.root.canvas scale all 0 0 $pyml_scale $pyml_scale
set x_position [expr {int($display_x + 0.5 * ($display_width - $pyml_width))}]
set y_position [expr {int($display_y + 0.5 * ($display_height - $pyml_height))}]
wm title . {PY-ML 启动提示}
set pyml_tick 0
proc pyml_animate {} {
    global pyml_tick pyml_scale
    set travel [expr {abs(($pyml_tick % 188) - 94) / 94.0}]
    set left [expr {(32 + 376 * $travel) * $pyml_scale}]
    .root.canvas coords loading $left [expr {173 * $pyml_scale}] [expr {$left + 120 * $pyml_scale}] [expr {182 * $pyml_scale}]
    incr pyml_tick
    after 30 pyml_animate
}
after 30 pyml_animate
