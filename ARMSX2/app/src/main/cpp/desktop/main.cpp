// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// Desktop Linux host layer for ARMSX2.
// Provides SDL3 window, VM lifecycle, and Host interface stubs.

#include "PrecompiledHeader.h"

#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

#include "SDL3/SDL.h"

#include "common/FileSystem.h"
#include "common/MemorySettingsInterface.h"
#include "common/Path.h"
#include "common/StringUtil.h"

#include "pcsx2/Achievements.h"
#include "pcsx2/GS.h"
#include "pcsx2/GSDumpReplayer.h"
#include "pcsx2/Host.h"
#include "pcsx2/ImGui/FullscreenUI.h"
#include "pcsx2/ImGui/ImGuiFullscreen.h"
#include "pcsx2/ImGui/ImGuiManager.h"
#include "pcsx2/Input/InputManager.h"
#include "pcsx2/MTGS.h"
#include "pcsx2/SIO/Pad/Pad.h"
#include "pcsx2/VMManager.h"
#include "pcsx2/GS/GSPerfMon.h"
#include "pcsx2/PerformanceMetrics.h"

static SDL_Window* s_window = nullptr;
static int s_window_width = 640;
static int s_window_height = 448;
static bool s_running = true;
static MemorySettingsInterface s_settings_interface;

// RunOnCPUThread queue
static std::mutex s_cpu_thread_mutex;
static std::deque<std::function<void()>> s_cpu_thread_queue;

// ---------------------------------------------------------------------------
// Host interface implementation
// ---------------------------------------------------------------------------

std::optional<WindowInfo> Host::AcquireRenderWindow(bool recreate_window)
{
	WindowInfo wi = {};
	wi.surface_width = s_window_width;
	wi.surface_height = s_window_height;
	wi.surface_scale = 1.0f;

	SDL_PropertiesID props = SDL_GetWindowProperties(s_window);
	void* x11_display = SDL_GetPointerProperty(props, "SDL.window.x11.display", nullptr);
	if (x11_display)
	{
		wi.type = WindowInfo::Type::X11;
		wi.display_connection = x11_display;
		wi.window_handle = (void*)(uintptr_t)SDL_GetNumberProperty(props, "SDL.window.x11.window", 0);
	}
	else
	{
		wi.type = WindowInfo::Type::Wayland;
		wi.display_connection = SDL_GetPointerProperty(props, "SDL.window.wayland.display", nullptr);
		wi.window_handle = SDL_GetPointerProperty(props, "SDL.window.wayland.surface", nullptr);
	}

	return wi;
}

void Host::ReleaseRenderWindow()
{
}

void Host::PumpMessagesOnCPUThread()
{
	// Drain RunOnCPUThread queue
	std::deque<std::function<void()>> queue;
	{
		std::lock_guard lock(s_cpu_thread_mutex);
		queue.swap(s_cpu_thread_queue);
	}
	for (auto& fn : queue)
		fn();

	// SDL event pump
	SDL_Event ev;
	while (SDL_PollEvent(&ev))
	{
		switch (ev.type)
		{
			case SDL_EVENT_QUIT:
				VMManager::SetState(VMState::Stopping);
				s_running = false;
				break;
			case SDL_EVENT_WINDOW_RESIZED:
				s_window_width = ev.window.data1;
				s_window_height = ev.window.data2;
				if (MTGS::IsOpen())
					MTGS::UpdateDisplayWindow();
				break;
			default:
				break;
		}
	}
}

void Host::RunOnCPUThread(std::function<void()> function, bool block)
{
	if (block)
	{
		std::mutex mtx;
		std::condition_variable cv;
		bool done = false;
		{
			std::lock_guard lock(s_cpu_thread_mutex);
			s_cpu_thread_queue.push_back([&]() {
				function();
				std::lock_guard lk(mtx);
				done = true;
				cv.notify_one();
			});
		}
		std::unique_lock lk(mtx);
		cv.wait(lk, [&] { return done; });
	}
	else
	{
		std::lock_guard lock(s_cpu_thread_mutex);
		s_cpu_thread_queue.push_back(std::move(function));
	}
}

void Host::RequestVMShutdown(bool allow_confirm, bool allow_save_state, bool default_save_state)
{
	VMManager::SetState(VMState::Stopping);
}

// --- BeginPresentFrame (GS stats, same as native-lib.cpp) ---

static s32 s_loop_count = 1;
static u32 s_dump_frame_number = 0;
static u32 s_loop_number = s_loop_count;
static double s_last_internal_draws = 0;
static double s_last_draws = 0;
static double s_last_render_passes = 0;
static double s_last_barriers = 0;
static double s_last_copies = 0;
static double s_last_uploads = 0;
static double s_last_readbacks = 0;
static u64 s_total_internal_draws = 0;
static u64 s_total_draws = 0;
static u64 s_total_render_passes = 0;
static u64 s_total_barriers = 0;
static u64 s_total_copies = 0;
static u64 s_total_uploads = 0;
static u64 s_total_readbacks = 0;
static u32 s_total_frames = 0;
static u32 s_total_drawn_frames = 0;

void Host::BeginPresentFrame()
{
	if (GSIsHardwareRenderer())
	{
		const u32 last_draws = s_total_internal_draws;
		const u32 last_uploads = s_total_uploads;

		static constexpr auto update_stat = [](GSPerfMon::counter_t counter, u64& dst, double& last) {
			const double val = g_perfmon.GetCounter(counter);
			dst += static_cast<u64>((val < last) ? val : (val - last));
			last = val;
		};

		update_stat(GSPerfMon::Draw, s_total_internal_draws, s_last_internal_draws);
		update_stat(GSPerfMon::DrawCalls, s_total_draws, s_last_draws);
		update_stat(GSPerfMon::RenderPasses, s_total_render_passes, s_last_render_passes);
		update_stat(GSPerfMon::Barriers, s_total_barriers, s_last_barriers);
		update_stat(GSPerfMon::TextureCopies, s_total_copies, s_last_copies);
		update_stat(GSPerfMon::TextureUploads, s_total_uploads, s_last_uploads);
		update_stat(GSPerfMon::Readbacks, s_total_readbacks, s_last_readbacks);

		const bool idle_frame = s_total_frames && (last_draws == s_total_internal_draws && last_uploads == s_total_uploads);

		if (!idle_frame)
			s_total_drawn_frames++;

		s_total_frames++;

		std::atomic_thread_fence(std::memory_order_release);
	}
}

// --- Trivial stubs (copied from native-lib.cpp) ---

void Host::OnGameChanged(const std::string& title, const std::string& elf_override, const std::string& disc_path,
	const std::string& disc_serial, u32 disc_crc, u32 current_crc)
{
}

void Host::CommitBaseSettingChanges()
{
}

void Host::LoadSettings(SettingsInterface& si, std::unique_lock<std::mutex>& lock)
{
}

void Host::CheckForSettingsChanges(const Pcsx2Config& old_config)
{
}

bool Host::RequestResetSettings(bool folders, bool core, bool controllers, bool hotkeys, bool ui)
{
	return false;
}

void Host::SetDefaultUISettings(SettingsInterface& si)
{
}

std::unique_ptr<ProgressCallback> Host::CreateHostProgressCallback()
{
	return nullptr;
}

void Host::ReportErrorAsync(const std::string_view title, const std::string_view message)
{
	if (!title.empty() && !message.empty())
		ERROR_LOG("ReportErrorAsync: {}: {}", title, message);
	else if (!message.empty())
		ERROR_LOG("ReportErrorAsync: {}", message);
}

void Host::OpenURL(const std::string_view url)
{
}

bool Host::CopyTextToClipboard(const std::string_view text)
{
	return false;
}

void Host::BeginTextInput()
{
}

void Host::EndTextInput()
{
}

std::optional<WindowInfo> Host::GetTopLevelWindowInfo()
{
	return std::nullopt;
}

void Host::OnInputDeviceConnected(const std::string_view identifier, const std::string_view device_name)
{
}

void Host::OnInputDeviceDisconnected(const InputBindingKey key, const std::string_view identifier)
{
}

void Host::SetMouseMode(bool relative_mode, bool hide_cursor)
{
}

void Host::RequestResizeHostDisplay(s32 width, s32 height)
{
}

void Host::OnVMStarting()
{
}

void Host::OnVMStarted()
{
}

void Host::OnVMDestroyed()
{
}

void Host::OnVMPaused()
{
}

void Host::OnVMResumed()
{
}

void Host::OnPerformanceMetricsUpdated()
{
}

void Host::OnSaveStateLoading(const std::string_view filename)
{
}

void Host::OnSaveStateLoaded(const std::string_view filename, bool was_successful)
{
}

void Host::OnSaveStateSaved(const std::string_view filename)
{
}

void Host::RefreshGameListAsync(bool invalidate_cache)
{
}

void Host::CancelGameListRefresh()
{
}

bool Host::IsFullscreen()
{
	return false;
}

void Host::SetFullscreen(bool enabled)
{
}

void Host::OnCaptureStarted(const std::string& filename)
{
}

void Host::OnCaptureStopped()
{
}

void Host::RequestExitApplication(bool allow_confirm)
{
}

void Host::RequestExitBigPicture()
{
}

void Host::OnAchievementsLoginSuccess(const char* username, u32 points, u32 sc_points, u32 unread_messages)
{
}

void Host::OnAchievementsLoginRequested(Achievements::LoginRequestReason reason)
{
}

void Host::OnAchievementsHardcoreModeChanged(bool enabled)
{
}

void Host::OnAchievementsRefreshed()
{
}

void Host::OnCoverDownloaderOpenRequested()
{
}

void Host::OnCreateMemoryCardOpenRequested()
{
}

bool Host::ShouldPreferHostFileSelector()
{
	return false;
}

void Host::OpenHostFileSelectorAsync(std::string_view title, bool select_directory, FileSelectorCallback callback,
	FileSelectorFilters filters, std::string_view initial_directory)
{
	callback(std::string());
}

void Host::ReportInfoAsync(const std::string_view title, const std::string_view message)
{
}

bool Host::LocaleCircleConfirm()
{
	return false;
}

bool Host::InNoGUIMode()
{
	return false;
}

int Host::LocaleSensitiveCompare(std::string_view lhs, std::string_view rhs)
{
	return lhs.compare(rhs);
}

// --- Stubs from AndroidStubs.cpp ---

BEGIN_HOTKEY_LIST(g_host_hotkeys)
END_HOTKEY_LIST()

void Host::SetMouseLock(bool state)
{
}

// --- InputManager stubs ---

std::optional<u32> InputManager::ConvertHostKeyboardStringToCode(const std::string_view str)
{
	return std::nullopt;
}

std::optional<std::string> InputManager::ConvertHostKeyboardCodeToString(u32 code)
{
	return std::nullopt;
}

const char* InputManager::ConvertHostKeyboardCodeToIcon(u32 code)
{
	return nullptr;
}

// --- Translation stubs ---

s32 Host::Internal::GetTranslatedStringImpl(
	const std::string_view context, const std::string_view msg, char* tbuf, size_t tbuf_space)
{
	if (msg.size() > tbuf_space)
		return -1;
	else if (msg.empty())
		return 0;

	std::memcpy(tbuf, msg.data(), msg.size());
	return static_cast<s32>(msg.size());
}

std::string Host::TranslatePluralToString(const char* context, const char* msg, const char* disambiguation, int count)
{
	TinyString count_str = TinyString::from_format("{}", count);

	std::string ret(msg);
	for (;;)
	{
		std::string::size_type pos = ret.find("%n");
		if (pos == std::string::npos)
			break;

		ret.replace(pos, pos + 2, count_str.view());
	}

	return ret;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char* argv0)
{
	std::fprintf(stderr, "Usage: %s [--app-root DIR] [--data-root DIR] <iso-path>\n", argv0);
}

int main(int argc, char* argv[])
{
	std::string iso_path;
	std::string app_root;
	std::string data_root;

	// Parse arguments
	for (int i = 1; i < argc; i++)
	{
		if (std::strcmp(argv[i], "--app-root") == 0 && i + 1 < argc)
			app_root = argv[++i];
		else if (std::strcmp(argv[i], "--data-root") == 0 && i + 1 < argc)
			data_root = argv[++i];
		else if (argv[i][0] != '-')
			iso_path = argv[i];
		else
		{
			print_usage(argv[0]);
			return 1;
		}
	}

	if (iso_path.empty())
	{
		print_usage(argv[0]);
		return 1;
	}

	// Default AppRoot: <source-tree>/app/src/main/assets
	// CMAKE_SOURCE_DIR is baked in at compile time.
	if (app_root.empty())
	{
#ifdef ARMSX2_SOURCE_DIR
		app_root = Path::Combine(ARMSX2_SOURCE_DIR, "../assets");
#else
		app_root = "../assets";
#endif
		if (!FileSystem::DirectoryExists(app_root.c_str()))
		{
			std::fprintf(stderr, "Cannot find assets dir at %s\n", app_root.c_str());
			std::fprintf(stderr, "Use --app-root to specify the assets directory.\n");
			return 1;
		}
	}

	// Default DataRoot: ~/.config/armsx2
	if (data_root.empty())
	{
		const char* xdg = std::getenv("XDG_CONFIG_HOME");
		if (xdg && xdg[0])
			data_root = Path::Combine(xdg, "armsx2");
		else
		{
			const char* home = std::getenv("HOME");
			if (home)
				data_root = Path::Combine(Path::Combine(home, ".config"), "armsx2");
			else
				data_root = "/tmp/armsx2";
		}
	}
	FileSystem::EnsureDirectoryExists(data_root.c_str(), false);

	EmuFolders::AppRoot = app_root;
	EmuFolders::DataRoot = data_root;
	EmuFolders::SetResourcesDirectory();

	Log::SetConsoleOutputLevel(LOGLEVEL_DEBUG);

	// Settings
	MemorySettingsInterface& si = s_settings_interface;
	Host::Internal::SetBaseSettingsLayer(&si);

	VMManager::SetDefaultSettings(si, true, true, true, true, true);

	si.SetBoolValue("EmuCore/GS", "FrameLimitEnable", false);
	si.SetIntValue("EmuCore/GS", "VsyncEnable", false);
	si.SetBoolValue("EmuCore", "EnableThreadPinning", true);
	si.SetBoolValue("EmuCore/CPU/Recompiler", "EnableFastmem", true);

	si.SetBoolValue("InputSources", "SDL", true);
	si.SetBoolValue("InputSources", "XInput", false);

	si.SetStringValue("SPU2/Output", "Backend", "SDL");
	si.SetBoolValue("EmuCore", "EnableFastBoot", false);
	si.SetIntValue("EmuCore/GS", "Renderer", static_cast<int>(GSRendererType::SW));

	Pad::ClearPortBindings(si, 0);
	si.ClearSection("Hotkeys");

	si.SetBoolValue("Logging", "EnableSystemConsole", true);
	si.SetBoolValue("Logging", "EnableTimestamps", true);
	si.SetBoolValue("Logging", "EnableVerbose", true);

	si.SetBoolValue("EmuCore/GS", "OsdShowFPS", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowSpeed", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowResolution", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowCPU", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowGPU", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowGSStats", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowFrameTimes", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowHardwareInfo", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowVersion", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowSettings", true);
	si.SetBoolValue("EmuCore/GS", "OsdShowInputs", true);

	VMManager::Internal::LoadStartupSettings();

	// SDL init
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD))
	{
		std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	s_window = SDL_CreateWindow("ARMSX2", s_window_width, s_window_height, SDL_WINDOW_RESIZABLE);
	if (!s_window)
	{
		std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		SDL_Quit();
		return 1;
	}

	// CPU thread init
	if (!VMManager::Internal::CPUThreadInitialize())
	{
		std::fprintf(stderr, "CPUThreadInitialize failed\n");
		VMManager::Internal::CPUThreadShutdown();
		SDL_DestroyWindow(s_window);
		SDL_Quit();
		return 1;
	}

	VMManager::ApplySettings();
	GSDumpReplayer::SetIsDumpRunner(false);

	VMBootParameters boot_params;
	boot_params.filename = iso_path;
	boot_params.fast_boot = false;

	Console.Error("Loading %s", iso_path.c_str());

	if (VMManager::Initialize(boot_params, nullptr) == VMBootResult::StartupSuccess)
	{
		Console.Error("VM INIT");
		VMManager::SetState(VMState::Running);

		while (s_running)
		{
			VMState state = VMManager::GetState();
			if (state == VMState::Stopping || state == VMState::Shutdown)
				break;
			else if (state == VMState::Running)
				VMManager::Execute();
			else
				usleep(250000);
		}

		VMManager::Shutdown(false);
	}
	else
	{
		std::fprintf(stderr, "VMManager::Initialize failed\n");
	}

	VMManager::Internal::CPUThreadShutdown();

	SDL_DestroyWindow(s_window);
	s_window = nullptr;
	SDL_Quit();

	return 0;
}
