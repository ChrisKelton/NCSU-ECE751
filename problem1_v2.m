N_trials = 1000;  %number of trials

run_simulation = true;
run_theoretical = true;
save_plot = true;
if ~run_simulation && ~run_theoretical
   fprintf("****ERROR: MUST TURN ON RUNNING SIMULATION AND/OR RUNNING THEORETICAL****\n")
   return
end
%%Simulation
%christopher kelton = i, o, e, e, o
deflection_ratio_colors = ["green", "blue", "red"];
if run_simulation
    % set up random seed & rng
    I = 256;
    O = 512;
    E = 64;
    seed = I + (2*O) + (2 * E);
    rnggnr = rng(seed, 'twister');
    
    deflection_ratios = [0.5, 1, 2];  % deflection ratios outlined by question in homework
    % want to keep m_input fixed, will just keep fixed at m = 1;
    % therefore, when we do our deflection ratios, we will be
    % varying the noise sigma (sigma_n)
    m = 1;
    number_of_gamma_ths = 20;
    % simulation_results will be stored as 
    % (each deflection_ratio, (Pf, Pd), (different gammas))
    simulation_results = zeros(length(deflection_ratios), 2, number_of_gamma_ths);
    
    for deflection_ratio_idx=1:length(deflection_ratios)   
       % determine gamma threshold for each change in deflection ratio
       % will simply be [deflection_ratio - 3, deflection_ratio + 3, number_of_gamma_ths]
       gamma_th_sim = linspace(deflection_ratios(deflection_ratio_idx)-3, deflection_ratios(deflection_ratio_idx)+3, number_of_gamma_ths);
       
       % deflection_ratio = m/std_n (std_n = sigma_n)
       % therefore, solve for sigma_n (different standard deviation for our
       % noise per deflection ratio)
       sigma_n = m / deflection_ratios(deflection_ratio_idx);
       for gamma_idx=1:length(gamma_th_sim)
           % get values for observations
           % observations
           % H_0: r = n
           % H_1: r = m + n

           n = sigma_n*randn(N_trials, 1);  % noise input with some variance sigma_n
           % H_0
           r_0 = n;
           false_alarms = sum(r_0 > gamma_th_sim(gamma_idx)) / N_trials;
           simulation_results(deflection_ratio_idx, 1, gamma_idx) = false_alarms;
           
           % H_1
           % want to use same random noise for r_1 for better simulation
           % comparison
           r_1 = m + n;
           detections = sum(r_1 > gamma_th_sim(gamma_idx)) / N_trials;
           simulation_results(deflection_ratio_idx, 2, gamma_idx) = detections;
       end
    end
    for idx=1:length(deflection_ratio_colors)
       scatter(squeeze(simulation_results(idx, 1, :)), squeeze(simulation_results(idx, 2, :)), deflection_ratio_colors(idx));
       hold on;
    end
end

%%Theoretical
if run_theoretical
    for d_idx=1:length(deflection_ratios)
        gamma_th = ((log(eta)/deflection_ratios(d_idx)) + (deflection_ratios(d_idx)/2));
        theoretical_false_alarms = (0.5)*erfc(gamma_th/sqrt(2));

        gamma_star_th = (gamma_th - deflection_ratios(d_idx))/sqrt(2);
        theoretical_detections = 0.5*erfc(gamma_star_th);

        plot(theoretical_false_alarms, theoretical_detections, deflection_ratio_colors(d_idx));
        hold on;
    end
    plot(linspace(0, 1), linspace(0, 1), '--');
end
title("ROC Curve")
xlabel("Pf");
ylabel("Pd");
legend('d=0.5', 'd=1', 'd=2');
legend('Location', 'southeast');
set(gcf, 'color', 'w');
if save_plot
	saveas(gcf, "ROC_Curve.png")
end