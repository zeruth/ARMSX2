// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "Host/AudioStream.h"

#include "common/Assertions.h"
#include "common/Console.h"
#include "common/Error.h"

#include "oboe/Oboe.h"

namespace {
	class OboeAudioStream final : public AudioStream,
	                               oboe::AudioStreamDataCallback,
	                               oboe::AudioStreamErrorCallback
	{
	public:
		OboeAudioStream(u32 sample_rate, const AudioStreamParameters& parameters);
		~OboeAudioStream() override;

		void SetPaused(bool paused) override;

		bool Initialize(bool stretch_enabled);
		bool Open();
		bool Start();
		void Stop();
		void Close();

		oboe::DataCallbackResult onAudioReady(oboe::AudioStream* p_audioStream,
			void* p_audioData, int32_t p_numFrames) override;
		bool onError(oboe::AudioStream* oboeStream, oboe::Result error) override;

	private:
		bool m_playing = false;
		bool m_stop_requested = false;

		std::shared_ptr<oboe::AudioStream> m_stream;
	};
} // namespace

oboe::DataCallbackResult OboeAudioStream::onAudioReady(oboe::AudioStream* p_audioStream,
	void* p_audioData, int32_t p_numFrames)
{
	if (p_audioData != nullptr)
		ReadFrames(reinterpret_cast<SampleType*>(p_audioData), p_numFrames);
	return oboe::DataCallbackResult::Continue;
}

bool OboeAudioStream::onError(oboe::AudioStream* oboeStream, oboe::Result error)
{
	Console.Error("(Oboe) ErrorCB %d", error);
	if (error == oboe::Result::ErrorDisconnected && !m_stop_requested)
	{
		Console.Error("(Oboe) Stream disconnected, reopening...");
		Stop();
		Close();
		if (!Open() || !Start())
			Console.Error("(Oboe) Failed to reopen stream after disconnection.");
		return true;
	}
	return false;
}

bool OboeAudioStream::Initialize(bool stretch_enabled)
{
	static constexpr const std::array<SampleReader, static_cast<size_t>(AudioExpansionMode::Count)> sample_readers = {{
		&StereoSampleReaderImpl,
		&SampleReaderImpl<AudioExpansionMode::StereoLFE,
			READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT, READ_CHANNEL_LFE>,
		&SampleReaderImpl<AudioExpansionMode::Quadraphonic,
			READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT,
			READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
		&SampleReaderImpl<AudioExpansionMode::QuadraphonicLFE,
			READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT, READ_CHANNEL_LFE,
			READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
		&SampleReaderImpl<AudioExpansionMode::Surround51,
			READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT, READ_CHANNEL_FRONT_CENTER,
			READ_CHANNEL_LFE, READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
		&SampleReaderImpl<AudioExpansionMode::Surround71,
			READ_CHANNEL_FRONT_LEFT, READ_CHANNEL_FRONT_RIGHT, READ_CHANNEL_FRONT_CENTER,
			READ_CHANNEL_LFE, READ_CHANNEL_SIDE_LEFT, READ_CHANNEL_SIDE_RIGHT,
			READ_CHANNEL_REAR_LEFT, READ_CHANNEL_REAR_RIGHT>,
	}};
	BaseInitialize(sample_readers[static_cast<size_t>(m_parameters.expansion_mode)], stretch_enabled);

	if (!Open())
		return false;
	if (!Start())
		return false;
	return true;
}

bool OboeAudioStream::Open()
{
	oboe::AudioStreamBuilder builder;
	builder.setDirection(oboe::Direction::Output);
	builder.setPerformanceMode(oboe::PerformanceMode::LowLatency);
	builder.setSharingMode(oboe::SharingMode::Shared);
	builder.setFormat(oboe::AudioFormat::Float);
	builder.setSampleRate(m_sample_rate);
	builder.setChannelCount(m_output_channels == 2 ? oboe::ChannelCount::Stereo : oboe::ChannelCount::Mono);
	builder.setDeviceId(oboe::kUnspecified);
	builder.setBufferCapacityInFrames(2048 * 2);
	builder.setFramesPerDataCallback(2048);
	builder.setDataCallback(this);
	builder.setErrorCallback(this);

	Console.WriteLn("(Oboe) Opening stream...");
	oboe::Result result = builder.openStream(m_stream);
	if (result != oboe::Result::OK)
	{
		Console.Error("(Oboe) openStream() failed: %d", result);
		return false;
	}
	return true;
}

bool OboeAudioStream::Start()
{
	if (m_playing)
		return true;

	Console.WriteLn("(Oboe) Starting stream...");
	m_stop_requested = false;

	oboe::Result result = m_stream->requestStart();
	if (result != oboe::Result::OK)
	{
		Console.Error("(Oboe) requestStart() failed: %d", result);
		return false;
	}
	m_playing = true;
	return true;
}

void OboeAudioStream::Stop()
{
	if (!m_playing)
		return;

	Console.WriteLn("(Oboe) Stopping stream...");
	m_stop_requested = true;

	oboe::Result result = m_stream->requestStop();
	if (result != oboe::Result::OK)
		Console.Error("(Oboe) requestStop() failed: %d", result);

	m_playing = false;
}

void OboeAudioStream::Close()
{
	Console.WriteLn("(Oboe) Closing stream...");
	if (m_playing)
		Stop();
	if (m_stream)
	{
		m_stream->close();
		m_stream.reset();
	}
}

void OboeAudioStream::SetPaused(bool paused)
{
	if (m_paused == paused)
		return;

	if (paused)
	{
		oboe::Result result = m_stream->requestPause();
		if (result != oboe::Result::OK)
		{
			Console.Error("(Oboe) requestPause() failed: %d", result);
			return;
		}
		m_playing = false;
	}
	else
	{
		Start();
	}
	m_paused = paused;
}

OboeAudioStream::OboeAudioStream(u32 sample_rate, const AudioStreamParameters& parameters)
	: AudioStream(sample_rate, parameters)
{
}

OboeAudioStream::~OboeAudioStream()
{
	Close();
}

std::unique_ptr<AudioStream> AudioStream::CreateOboeAudioStream(u32 sample_rate,
	const AudioStreamParameters& parameters, bool stretch_enabled, Error* error)
{
	std::unique_ptr<OboeAudioStream> stream = std::make_unique<OboeAudioStream>(sample_rate, parameters);
	if (!stream->Initialize(stretch_enabled))
		stream.reset();
	return stream;
}
