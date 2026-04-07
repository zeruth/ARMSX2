package com.armsx2.ui.toolbar

import android.content.Intent
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.vector.ImageVector
import com.armsx2.Main
import compose.icons.LineAwesomeIcons
import compose.icons.lineawesomeicons.FolderOpen

class LoadGameButton : ToolbarButton() {
    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.FolderOpen)
    override fun isVisible(): Boolean {
        return true
    }

    override fun action() {
        // Internal storage
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, false)
        intent.setType("*/*")
        Main.instance?.openFileAction?.launch(intent)
    }
}