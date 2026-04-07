package com.armsx2.ui.toolbar

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.vector.ImageVector
import com.armsx2.EmuState
import com.armsx2.Main
import compose.icons.LineAwesomeIcons
import compose.icons.lineawesomeicons.PauseSolid

class PauseButton : ToolbarButton() {
    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.PauseSolid)
    override fun isVisible(): Boolean {
        return Main.eState.value == EmuState.RUNNING
    }

    override fun action() {
        Main.pause()
    }
}