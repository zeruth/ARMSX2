package com.pasx2.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import com.pasx2.EmuState
import com.pasx2.Main

object WindowImpl {
    val toolbarVisible = mutableStateOf(true)
    @Composable
    fun Window(content: @Composable () -> Unit) {
        //Container
        Box(Modifier.fillMaxSize().background(Color.DarkGray)) {
            //Container Layout
            Row(Modifier.fillMaxSize()) {
                //Game View
                Box(Modifier.weight(1f).background(Color.Transparent)) {
                    content.invoke()
                }
                if (Main.eState.value != EmuState.RUNNING || toolbarVisible.value)
                    ToolbarImpl.Toolbar()
            }
        }
    }
}