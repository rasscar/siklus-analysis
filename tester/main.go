package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func main() {
	r := bufio.NewReader(os.Stdin)
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()

	var t, tc, mode int
	var jumLolos, jumGagal int
	var errBaca error

	fmt.Fscan(r, &t)
	jumLolos = 0
	jumGagal = 0

	w.WriteString("============================================================\n")
	w.WriteString("  SIKLUS ML TEST HARNESS (GO I/O RUNNER + PYTHON ENGINE)\n")
	w.WriteString("============================================================\n")

	for tc = 1; tc <= t; tc++ {
		fmt.Fscan(r, &mode)

		var con bool
		con = false

		if mode == 1 {
			var prevDur, sleep, stress, deltaBmi, junk float64
			var expDur float64
			var expOvul, expStart, expEnd int

			fmt.Fscan(r, &prevDur, &sleep, &stress, &deltaBmi, &junk)
			fmt.Fscan(r, &expDur, &expOvul, &expStart, &expEnd)

			var inputStr string
			inputStr = fmt.Sprintf("1 %f %f %f %f %f", prevDur, sleep, stress, deltaBmi, junk)

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(inputStr)

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 1] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actDur float64
			var actOvul, actStart, actEnd int
			fmt.Sscan(string(outBytes), &actDur, &actOvul, &actStart, &actEnd)

			var bedaDur float64
			bedaDur = actDur - expDur
			if bedaDur < 0 {
				bedaDur = -bedaDur
			}

			if bedaDur <= 0.5 && actOvul == expOvul && actStart == expStart && actEnd == expEnd {
				con = true
			}

			if con {
				fmt.Fprintf(w, "Kasus #%d: [MODE 1 - RIDGE CYCLE] LOLOS\n", tc)
				fmt.Fprintf(w, "  Prediksi: %.2f hari | Ovulasi: hari ke-%d | Subur: (%d-%d)\n", actDur, actOvul, actStart, actEnd)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 1 - RIDGE CYCLE] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: durasi=%.2f ovul=%d rentang=(%d-%d)\n", actDur, actOvul, actStart, actEnd)
				fmt.Fprintf(w, "  Harap: durasi=%.2f ovul=%d rentang=(%d-%d)\n", expDur, expOvul, expStart, expEnd)
				jumGagal = jumGagal + 1
			}

		} else if mode == 2 {
			var cycleDur float64
			var dayOffset int
			var stress, sleep float64
			var expProb float64
			var expPeak int

			fmt.Fscan(r, &cycleDur, &dayOffset, &stress, &sleep)
			fmt.Fscan(r, &expProb, &expPeak)

			var inputStr string
			inputStr = fmt.Sprintf("2 %f %d %f %f", cycleDur, dayOffset, stress, sleep)

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(inputStr)

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 2] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actProb float64
			var actPeak int
			fmt.Sscan(string(outBytes), &actProb, &actPeak)

			var bedaProb float64
			bedaProb = actProb - expProb
			if bedaProb < 0 {
				bedaProb = -bedaProb
			}

			if bedaProb <= 0.05 && actPeak == expPeak {
				con = true
			}

			if con {
				var statusPuncak string
				if actPeak == 1 {
					statusPuncak = "PEAK FERTILITY"
				} else {
					statusPuncak = "STANDAR"
				}
				fmt.Fprintf(w, "Kasus #%d: [MODE 2 - LOGISTIC PEAK] LOLOS\n", tc)
				fmt.Fprintf(w, "  Peluang: %.4f | Status: %s\n", actProb, statusPuncak)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 2 - LOGISTIC PEAK] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: p=%.4f peak=%d | Harap: p=%.4f peak=%d\n", actProb, actPeak, expProb, expPeak)
				jumGagal = jumGagal + 1
			}

		} else if mode == 3 {
			var bmiPra, beratPra float64
			var week int
			var beratAktual float64
			var expIdeal, expDev float64
			var expStatus int

			fmt.Fscan(r, &bmiPra, &beratPra, &week, &beratAktual)
			fmt.Fscan(r, &expIdeal, &expDev, &expStatus)

			var inputStr string
			inputStr = fmt.Sprintf("3 %f %f %d %f", bmiPra, beratPra, week, beratAktual)

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(inputStr)

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 3] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actIdeal, actDev float64
			var actStatus int
			fmt.Sscan(string(outBytes), &actIdeal, &actDev, &actStatus)

			var bedaIdeal, bedaDev float64
			bedaIdeal = actIdeal - expIdeal
			if bedaIdeal < 0 {
				bedaIdeal = -bedaIdeal
			}
			bedaDev = actDev - expDev
			if bedaDev < 0 {
				bedaDev = -bedaDev
			}

			if bedaIdeal <= 0.2 && bedaDev <= 0.2 && actStatus == expStatus {
				con = true
			}

			if con {
				var labelStatus string
				if actStatus == 0 {
					labelStatus = "NORMAL"
				} else if actStatus == 1 {
					labelStatus = "DI ATAS IDEAL"
				} else {
					labelStatus = "DI BAWAH IDEAL"
				}
				fmt.Fprintf(w, "Kasus #%d: [MODE 3 - GWG SPLINE] LOLOS\n", tc)
				fmt.Fprintf(w, "  Ideal: %.2f kg | Deviasi: %+.2f kg | Status: %s\n", actIdeal, actDev, labelStatus)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 3 - GWG SPLINE] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: ideal=%.2f dev=%.2f stat=%d\n", actIdeal, actDev, actStatus)
				fmt.Fprintf(w, "  Harap: ideal=%.2f dev=%.2f stat=%d\n", expIdeal, expDev, expStatus)
				jumGagal = jumGagal + 1
			}

		} else if mode == 4 {
			var predDur, prevDur, sleep, stress, deltaBmi, junk float64
			var expRisk int
			var expFactor string

			fmt.Fscan(r, &predDur, &prevDur, &sleep, &stress, &deltaBmi, &junk)
			fmt.Fscan(r, &expRisk, &expFactor)

			var inputStr string
			inputStr = fmt.Sprintf("4 %f %f %f %f %f %f", predDur, prevDur, sleep, stress, deltaBmi, junk)

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(inputStr)

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 4] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actRisk int
			var actFactor string
			var actDelta float64
			fmt.Sscan(string(outBytes), &actRisk, &actFactor, &actDelta)

			if actRisk == expRisk && actFactor == expFactor {
				con = true
			}

			if con {
				var labelRisiko string
				if actRisk == 0 {
					labelRisiko = "NORMAL"
				} else if actRisk == 1 {
					labelRisiko = "SEDANG"
				} else {
					labelRisiko = "TINGGI"
				}
				fmt.Fprintf(w, "Kasus #%d: [MODE 4 - XAI RISK] LOLOS\n", tc)
				fmt.Fprintf(w, "  Level: %s | Faktor Utama: %s (%+.2f hari)\n", labelRisiko, actFactor, actDelta)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 4 - XAI RISK] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: risiko=%d faktor=%s\n", actRisk, actFactor)
				fmt.Fprintf(w, "  Harap: risiko=%d faktor=%s\n", expRisk, expFactor)
				jumGagal = jumGagal + 1
			}

		} else if mode == 5 {
			var numDays, i int
			var expConf, expOvul int
			var expDelta float64

			fmt.Fscan(r, &numDays)
			var suhuArr []float64
			suhuArr = make([]float64, numDays)
			for i = 0; i < numDays; i++ {
				fmt.Fscan(r, &suhuArr[i])
			}
			fmt.Fscan(r, &expConf, &expOvul, &expDelta)

			var sb strings.Builder
			sb.WriteString("5 ")
			sb.WriteString(strconv.Itoa(numDays))
			for i = 0; i < numDays; i++ {
				sb.WriteString(" ")
				sb.WriteString(fmt.Sprintf("%f", suhuArr[i]))
			}

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(sb.String())

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 5] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actConf, actOvul int
			var actDelta float64
			fmt.Sscan(string(outBytes), &actConf, &actOvul, &actDelta)

			var bedaDelta float64
			bedaDelta = actDelta - expDelta
			if bedaDelta < 0 {
				bedaDelta = -bedaDelta
			}

			if actConf == expConf && actOvul == expOvul && bedaDelta <= 0.05 {
				con = true
			}

			if con {
				var statusOvul string
				if actConf == 1 {
					statusOvul = "OVULASI TERKONFIRMASI"
				} else {
					statusOvul = "BELUM TERKONFIRMASI"
				}
				fmt.Fprintf(w, "Kasus #%d: [MODE 5 - BBT CUSUM] LOLOS\n", tc)
				fmt.Fprintf(w, "  Status: %s | Hari ke-%d | Lonjakan: %+.2f C\n", statusOvul, actOvul, actDelta)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 5 - BBT CUSUM] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: conf=%d ovul=%d delta=%.2f\n", actConf, actOvul, actDelta)
				fmt.Fprintf(w, "  Harap: conf=%d ovul=%d delta=%.2f\n", expConf, expOvul, expDelta)
				jumGagal = jumGagal + 1
			}

		} else if mode == 6 {
			var diet, stress, sleep, sugar float64
			var expProb float64
			var expCode int

			fmt.Fscan(r, &diet, &stress, &sleep, &sugar)
			fmt.Fscan(r, &expProb, &expCode)

			var inputStr string
			inputStr = fmt.Sprintf("6 %f %f %f %f", diet, stress, sleep, sugar)

			var cmd *exec.Cmd
			cmd = exec.Command("python", "engine.py")
			cmd.Stdin = strings.NewReader(inputStr)

			var outBytes []byte
			outBytes, errBaca = cmd.Output()
			if errBaca != nil {
				fmt.Fprintf(w, "Kasus #%d: [MODE 6] ERROR EKSEKUSI PYTHON\n", tc)
				jumGagal = jumGagal + 1
				continue
			}

			var actProb float64
			var actCode int
			fmt.Sscan(string(outBytes), &actProb, &actCode)

			var bedaProb float64
			bedaProb = actProb - expProb
			if bedaProb < 0 {
				bedaProb = -bedaProb
			}

			if bedaProb <= 0.05 && actCode == expCode {
				con = true
			}

			if con {
				var labelStatus string
				if actCode == 0 {
					labelStatus = "RENDAH (NORMAL)"
				} else if actCode == 1 {
					labelStatus = "SEDANG (WASPADA)"
				} else {
					labelStatus = "TINGGI (RESIKO OLIGOMENOREA)"
				}
				fmt.Fprintf(w, "Kasus #%d: [MODE 6 - REGULARITY RISK] LOLOS\n", tc)
				fmt.Fprintf(w, "  Peluang Risiko: %.4f | Klasifikasi: %s\n", actProb, labelStatus)
				jumLolos = jumLolos + 1
			} else {
				fmt.Fprintf(w, "Kasus #%d: [MODE 6 - REGULARITY RISK] GAGAL\n", tc)
				fmt.Fprintf(w, "  Dapat: p=%.4f code=%d | Harap: p=%.4f code=%d\n", actProb, actCode, expProb, expCode)
				jumGagal = jumGagal + 1
			}
		}
	}

	w.WriteString("------------------------------------------------------------\n")
	w.WriteString("RINGKASAN:\n")
	w.WriteString("Total Kasus: " + strconv.Itoa(t) + "\n")
	w.WriteString("Lolos      : " + strconv.Itoa(jumLolos) + "\n")
	w.WriteString("Gagal      : " + strconv.Itoa(jumGagal) + "\n")
	w.WriteString("============================================================\n")
}
