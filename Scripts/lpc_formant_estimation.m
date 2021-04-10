function [formants] = lpc_formant_estimation(ar, Fs, lower_bound, bw_bound)
    % Estimate the frequencies of the formants from the roots of the LPC
    % polynomial given by the AR coefficients 
    
    % Find the roots of the prediction polynomial returned by lpc
    rts = roots(ar);
    % Because the LPC coefficients are real-valued, the roots occur in complex conjugate pairs. 
    % Retain only the roots with one sign for the imaginary part and determine the angles corresponding to the roots.
    rts = rts(imag(rts)>=0);
    angz = atan2(imag(rts), real(rts));

    % Convert frequency angles to Hz and sort
    [frqs,indices] = sort(angz.*(Fs/(2*pi)));
    
    % The bandwidths of the formants are represented by the distance of the roots from the unit circle.
    bw = -1/2*(Fs/(2*pi))*log(abs(rts(indices)));

    % Return formant frequencies subject to conditions
    nn = 1;
    for kk = 1:length(frqs)
        if (frqs(kk) > lower_bound && bw(kk) < bw_bound)
            formants(nn) = frqs(kk);
            nn = nn+1;
        end
    end
 end

