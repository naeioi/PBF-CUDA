#pragma once
#define LOG_MEMB(s) double s##_bg, s##_sum, s##_min, s##_max

class Logger
{
public:
	enum TType {
		FRAME_START,
		SIMULATE_START,
		SIMULATE_END,
		RENDER_START,
		RENDER_END,
		ADVECT_START, ADVECT_END,
		GRID_START, GRID_END,
		DENSITY_START, DENSITY_END,
		VELOCITY_UPDATE_START, VELOCITY_UPDATE_END,
		VELOCITY_CORRECT_START, VELOCITY_CORRECT_END,
		DEPTH_START, DEPTH_END,
		THICK_START, THICK_END,
		SMOOTH_START, SMOOTH_END,
		NORMAL_START, NORMAL_END,
		SHADING_START, SHADING_END
	};

	Logger();
	~Logger();

	static Logger& getInstance();
	void toggleLogTime(bool state) { enableLogTime = state; }
	void logTime(TType type);
	void report();

private:
	static Logger * m_instance;

	int nframe;
	bool enableLogTime;

	LOG_MEMB(simu);
	LOG_MEMB(render);
	LOG_MEMB(advect);
	LOG_MEMB(grid);
	LOG_MEMB(density);
	LOG_MEMB(vel_upd);
	LOG_MEMB(vel_corr);
	LOG_MEMB(depth);
	LOG_MEMB(thick);
	LOG_MEMB(smooth);
	LOG_MEMB(normal);
	LOG_MEMB(shading);
};

