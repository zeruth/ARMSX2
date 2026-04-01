package com.pasx2.ui.toolbar

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import com.pasx2.EmuState
import com.pasx2.Main
import compose.icons.AllIcons
import compose.icons.FontAwesomeIcons
import compose.icons.LineAwesomeIcons
import compose.icons.fontawesomeicons.Regular
import compose.icons.fontawesomeicons.Solid
import compose.icons.fontawesomeicons.regular.PlayCircle
import compose.icons.fontawesomeicons.solid.Play
import compose.icons.lineawesomeicons.ArrowCircleUpSolid
import compose.icons.lineawesomeicons.CheckCircleSolid
import compose.icons.lineawesomeicons.PlaySolid
import compose.icons.lineawesomeicons.RedoAltSolid
import compose.icons.lineawesomeicons.RedoSolid

class RestartButton : ToolbarButton() {
    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.RedoAltSolid)
    override var background = mutableStateOf(Color.Red)
    override fun isVisible(): Boolean {
        return Main.eState.value == EmuState.RUNNING
    }

    override fun action() = Main.restart()
}