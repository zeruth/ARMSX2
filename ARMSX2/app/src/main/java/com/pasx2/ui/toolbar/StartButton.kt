package com.pasx2.ui.toolbar

import androidx.compose.runtime.mutableStateOf
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
import compose.icons.lineawesomeicons.PlaySolid
import compose.icons.lineawesomeicons.PlugSolid

class StartButton : ToolbarButton() {
    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.PlaySolid)
    override fun isVisible(): Boolean {
        return Main.eState.value != EmuState.RUNNING
    }

    override fun action() {
        when (Main.eState.value) {
            EmuState.STOPPED -> Main.start()
            EmuState.PAUSED -> Main.resume()
            else -> {}
        }
    }
}